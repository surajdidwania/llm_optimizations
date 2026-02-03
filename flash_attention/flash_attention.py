"""
Flash Attention is memory efficient attention for multi head attention
- Uses tiling mechanism
- Uses incrementally softmax mechanism
- Faster training and inference of the transformer models
"""
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch


