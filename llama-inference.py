import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math
import torch.nn as nn
from typing import Optional

@dataclass
class LlamaConfig:
    n_emb: int = 4098
    n_layers: int = 32
    vocab_size: int = -1
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplication: Optional[float] = None
    norm_eps: float = 1e-5
    #for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device:str = None

class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        assert config.vocab_size != -1
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(self.vocab_size,config.n_emb)
        self.n_layers = config.n_layers
        self.layers = nn.ModuleList([BLock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.n_emb,eps=config.norm_eps)
        self.output = nn.Linear(config.n_emb,self.vocab_size,bias=False)
        self.frequency_complex = compute_theta_pos_freq(self.config.n_emb//self.config.n_heads,self.config.max_seq_len * 2,device=self.config.device)