import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch


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
        :param token_index:
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

    @torch.no_grad()
    def forward(
            self,
            query: torch.Tensor,
            sequence: Sequence,
            query_start_pos: Optional[int] = None,
    ) -> torch.Tensor:

        assert query.ndim == 3, "query must be (Tq, H, D)"
        Tq, H, D = query.shape
        assert H == self.num_heads and D == self.head_dim

        Tkv = sequence.get_num_tokens() # this checks how many tokens exist in the sequence so far, whose K/V are in the cache?
        if query_start_pos is None:
            query_start_pos = Tkv - Tq
        assert 0<= query_start_pos <= Tkv - Tq

        # streaming softmax state per (Tq, H)
        m = torch.full((Tq, H), -float("inf"), device=query.device, dtype = query.dtype) # running max
        l = torch.zeros((Tq, H), device=query.device, dtype = query.dtype) # running sum exp
        out = torch.zeros((Tq, H, D), device=query.device, dtype=query.dtype) # running weighted sum

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

            K_h = K.permute(1, 0, 2) # (H, Tk, D)
            V_h = V.permute(1, 0, 2) # (H, Tk, D)

            # scores: (Tq, H, Tk)
            scores = torch.einsum("thd,hkd->thk", query, K_h) * self.scale

            # causal mask: key_pos <= query_pos
            k_pos = torch.arange(page_start, page_start+valid_len, device=query.device)
            causal = (k_pos[None, :] <= q_pos[:, None])
            scores = scores.masked_fill(~causal[:, None, :], -float("inf")) # setting - infinity for future tokens

            #streaming softmax update
            scores_max = scores.max(dim=-1).values
            new_m = torch.maximum(m, scores_max)
            exp_scores = torch.exp(scores - new_m[..., None])
            exp_m = torch.exp(m - new_m)

            new_l = l * exp_m + exp_scores.sum(dim=-1)
            weighted = torch.einsum("thk,hkd->thd", exp_scores, V_h)
            new_out = out * exp_m[..., None] + weighted

            m, l, out = new_m, new_l, new_out

        l = torch.clamp(l, min=1e-9)
        return out / l[..., None] #normalized ie numerator = weighted sum of values and denominator = sum of weights as we calculate softmax



if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = Config(num_heads=8, head_dim=64, num_pages=10, page_size=16)
    bm = BlockManager(cfg.num_pages, cfg.page_size, cfg.num_heads, cfg.head_dim)
    
    prompt_tokens = list(range(20))
    seq = Sequence(seq_id=1, prompt_tokens=prompt_tokens, page_size=cfg.page_size)

    # Allocate for prefill
    ok = bm.allocate_for_sequence(seq)
    print(f"Prefill allocation ok: {ok} | {seq}")

    # Write KV for all prompt tokens
    for t in range(seq.get_num_tokens()):
        k = torch.randn(cfg.num_heads, cfg.head_dim)
        v = torch.randn(cfg.num_heads, cfg.head_dim)
        seq.set_kv_at(t, k, v)

    # Decode: append tokens and write KV per token
    for i in range(20):
        seq.append_tokens(100 + i)
        ok = bm.allocate_for_sequence(seq)
        assert ok, "out of pages"

        t = seq.get_num_tokens()-1
        k = torch.randn(cfg.num_heads, cfg.head_dim)
        v = torch.randn(cfg.num_heads, cfg.head_dim)
        seq.set_kv_at(t, k, v)

    print("after decode:", seq, "| free_pages:", bm.get_num_free_pages())

    pa = PagedAttention(cfg.num_heads, cfg.head_dim)
    q = torch.randn(1, cfg.num_heads, cfg.head_dim)
    attention_output = pa.forward(q, seq)
    print(f"\n Paged Attention output shape is {attention_output.shape}")
    