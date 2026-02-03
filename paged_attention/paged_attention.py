import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from transformers.utils.fx import torch_cat


@dataclass
class Config:
    num_heads: int = 8
    head_dim: int = 64
    num_pages: int = 10
    page_size: int = 16
    
    
@dataclass
class Page:
    page_size: int
    num_heads: int
    head_dim: int
    ref_count: int = 0
    hash: str = field(default_factory= lambda: uuid.uuid4().hex)

    def __post_init__(self):
        # kv: (2, page_size, num_heads, head_dim)
        self.kv = torch.zeros(2, self.page_size, self.num_heads, self.head_dim)

    def __repr__(self):
        return f"{self.hash} | page_size={self.page_size} | ref_count={self.ref_count}"

    def __hash__(self):
        return hash(self.hash)


@dataclass
class PageTable:
    # maps logical page id to physical page
    map: dict = field(default_factory=dict)

    def map_page(self, logical_id: int, physical_page: Page) -> None:
        self.map[logical_id] = physical_page

    def get_page(self, logical_id: int) -> Optional[Page]:
        return self.map.get(logical_id)


@dataclass
class Sequence:
    """
    tokens: prompt + generated tokens. We keep a page table mapping and logical pages list
    """
    def __init__(self, seq_id: int, prompt_tokens: List[int], page_size: int) -> None:
        self.seq_id = seq_id
        self.tokens = prompt_tokens.copy()
        self.page_size = page_size
        self.page_table = PageTable()
        self.logical_pages: List[int] = []

    def __repr__(self):
        return f"Sequence(id={self.seq_id}, tokens={len(self.tokens)}, pages={len(self.logical_pages)})"

    def get_num_tokens(self) -> int:
        return len(self.tokens)

    def get_num_pages_needed(self) -> int:
        return (len(self.tokens) + self.page_size-1) // self.page_size

    def append_tokens(self, token_id: int) -> None:
        self.tokens.append(token_id)

    def token_to_page_offset(self, token_index: int) -> Tuple[int, int]:
        """
        :return: (logical_page_id, offset_in_page) for a token index in this sequence.
        """
        return token_index // self.page_size, token_index % self.page_size

    def set_kv_at(self, token_index: int, k: torch.Tensor, v: torch.Tensor):
        """
        :param k: (num_heads, head_dim)
        :param v:(num_heads, head_dim)
        :return: Writes KV for the token at the right page/offset
        """
        logical_id, offset = self.token_to_page_offset(token_index)
        page = self.page_table.get_page(logical_id)
        if page is None:
            raise RuntimeError(f"No Page mapped for logical_id={logical_id}")

        page.kv[0, offset].copy_(k)
        page.kv[1, offset].copy_(v)

class BlockManager:
    def __init__(self, num_pages: int, page_size: int, num_heads: int, head_dim: int) -> None:
        self.pages = [Page(page_size, num_heads, head_dim) for _ in range(num_pages)]
        self.free = deque(self.pages)
        self.allocated = set()

    def _allocate(self) -> Optional[Page]:
        if not self.free:
            return None

        page_to_be_allocated = self.free.popleft()
        page_to_be_allocated.ref_count += 1
        self.allocated.add(page_to_be_allocated)
        return page_to_be_allocated

    def _deallocate(self, page: Page):
        if page not in self.allocated:
            return

        page.ref_count -=1
        if page.ref_count==0:
            self.allocated.remove(page)
            self.free.append(page)
            page.kv.zero_()

    def allocate_for_sequence(self, sequence: Sequence):
        needed = sequence.get_num_pages_needed()
        current_pages = len(sequence.logical_pages)
        to_allocate = needed - current_pages

        if to_allocate <=0:
            return True
        if len(self.free) < to_allocate:
            return False

        for idx in range(to_allocate):
            page = self._allocate()
            if page is None:
                return False

            logical_id = current_pages + idx
            sequence.logical_pages.append(logical_id)
            sequence.page_table.map_page(logical_id, page)

        return True

    def free_sequence(self, sequence: Sequence):
        for logical_id in sequence.logical_pages:
            page = sequence.page_table.get_page(logical_id)
            if page is not None:
                self._deallocate(page)

        sequence.logical_pages.clear()
        sequence.page_table.map.clear()

    def get_num_free_pages(self) -> int:
        return len(self.free)


class PagedAttention:
    """
    Computes attention without concatenating KV:
     - Iterates over pages
     - streaming softmax accumulations
     - supports causal masking
     - Tkv: total number of tokens that already have K/V stored
    """
    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

    @torch.no_grad
    def forward(
            self,
            query: torch.Tensor,
            sequence: Sequence,
            query_start_pos: Optional[int] = None,
    ) -> torch.Tensor:
        assert query.ndim == 3, "query must be (Tq, H, D)"
        Tq, H, D = query.shape
        assert H == self.num_heads and D == self.head_dim

        Tkv = sequence.get_num_tokens() # How many tokens exist in the sequence so far, whose K/V are in the cache?
        if query_start_pos is None:
            query_start_pos = Tkv - Tq
        assert 0<= query_start_pos <= Tkv - Tq

        k_blocks, v_blocks = [], []
        for logical_id in sequence.logical_pages:
            page = sequence.page_table.get_page(logical_id)
            k_blocks.append(page.kv[0])
            v_blocks.append(page.kv[1])

        # streaming softmax state per (Tq, H)
        m = torch.full((Tq, H), -float("inf"), device=query.device, dtype = query.dtype)
        l = torch.zeros((Tq, H), device=query.device, dtype = query.dtype)
        out = torch.zeros((Tq, H, D), device=query.device, dtype=query.dtype)

        # Absolute position for each query token
        q_pos = torch.arange(query_start_pos, query_start_pos + Tq, device=query.device)

        for logical_id in sequence.logical_pages:
            page = sequence.page_table.get_page(logical_id)
            if page is None:
                continue

            page_start = logical_id * sequence.page_size
            if page_start >= Tkv:
                break

            valid_len = min(sequence.page_size, Tkv - page_start)
            if valid_len <=0:
                continue

            K = page.kv[0, :valid_len].to(query.device, query.dtype)  # (Tk, H, D)
            V = page.kv[1, :valid_len].to(query.device, query.dtype) # (Tk, H, D)



        # keys = torch.cat(k_blocks, dim=0) # (tokens, num_head, head_dim)
        # values = torch.cat(v_blocks, dim=0)
        #
        # num_tokens = sequence.get_num_tokens()
        # keys = keys[:num_tokens].transpose(0,1)
        # values = values[:num_tokens].transpose(0,1)
        #
        # q = query.transpose(0,1)
        # attention_score = torch.matmul(q, keys.transpose(1,2 )) * self.scale
        # attention_weights = F.softmax(attention_score, dim=-1)
        # output = torch.matmul(attention_weights, values)
        #
        # return output.transpose(0,1)


if __name__ == "__main__":
    config = Config(num_head=8, head_dim=64, page_size=16)
    block_manager = BlockManager(num_pages=10, page_size=config.page_size)
    
    print(f"Initialized block manager with {len(block_manager.pages)} pages")
    print(f"Page size: {config.page_size} tokens per page")
    
    prompt_tokens = list(range(50))
    seq1 = Sequence(seq_id=1, prompt_tokens=prompt_tokens, page_size=config.page_size)
    
    print(f"\n {seq1}")
    print(f"tokens: {seq1.get_num_tokens()}")
    print(f"pages needed: {seq1.get_num_pages_needed()}")

    print(f"=======PREFILL PHASE======")
    success = block_manager.allocate_for_sequence(seq1)
    print(f"allocation successful: {success}")
    print(f"logical phase: {seq1.logical_pages}")
    print(f"page table: {seq1.page_table}")
    print(f"Free pages remaining: {block_manager.get_num_free_pages()}")

    print(f"=======DECODE PHASE======")
    for i in range(20):
        new_token = 100 + i
        seq1.append_tokens(new_token)

        pages_needed = seq1.get_num_pages_needed()
        if pages_needed > len(seq1.logical_pages):
            print(f"\ntoken {i + 1}: need new page (total tokens: {seq1.get_num_tokens()})")
            success = block_manager.allocate_for_sequence(seq1)
            print(f"allocated page {seq1.logical_pages[-1]}")
        else:
            print(f"token {i + 1}: using existing pages (total tokens: {seq1.get_num_tokens()})")

    print(f"\nfinal sequence state: {seq1}")
    print(f"free pages: {block_manager.get_num_free_pages()}")

    pa = PagedAttention(config.num_head, config.head_dim)
    mock_query = torch.randn(1, config.num_head, config.head_dim)

    for log_id in seq1.logical_pages:
        page = seq1.page_table.get_page(log_id)
        page.kv.uniform_(-1, 1)

    attention_output = pa.forward(mock_query, seq1)
    print(f"\n Paged Attention output shape is {attention_output.shape}")
    