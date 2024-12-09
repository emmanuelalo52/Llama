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

def repeat_kv(x, n_rep):
    batch_size,seq_len,n_kv_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (x[:,:,:,None,:].expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim).reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim))

class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_head_q = config.n_heads
        self.r_rep = self.n_head_q//self.n_kv_heads #split the ratio of queries and kv
        self.head_dim = config.n_emb//config.n_heads
        #weights and projection
        self.wq = nn.Linear(config.n_emb,config.n_heads*self.head_dim,bias=False)
        self.wk = nn.Linear(config.n_emb,self.n_kv_heads*self.head_dim,bias=False)
        self.wv = nn.Linear(config.n_emb,self.n_kv_heads*self.head_dim,bias=False)
        self.wo = nn.Linear(config.n_heads*self.head_dim,config.n_emb,bias=False)

        #cache for k and v
        self.k_cache = torch.zeros((config.max_batch_size,config.max_seq_len,self.n_kv_heads,self.head_dim))
        self.v_cache = torch.zeros((config.max_batch_size,config.max_seq_len,self.n_kv_heads,self.head_dim))

    def forward(self,x,start_pos,frequency_complex):
        batch_size,seq_len,_ = x.shape

        query_weight = self.wq(x)
        key_weight = self.wk(x)
        value_weight = self.wv(x)

        query_weight = query_weight.view(batch_size,seq_len,self.n_head_q,self.head_dim)
        key_weight = key_weight.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)

        value_weight = value_weight.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)

        #apply RoPE
        query_weight = rotatry_postional_embedding(query_weight,frequency_complex,device=x.device)
        key_weight = rotatry_postional_embedding(key_weight,frequency_complex,device=x.device)

        #append the position of kv cache
        self.k_cache[:batch_size,start_pos:start_pos+seq_len] = key_weight
        self.v_cache[:batch_size,start_pos:start_pos+seq_len] = value_weight

        #retrieve the tokens into the kv cache
        keys = self.k_cache[:batch_size,0:start_pos+seq_len]
        value = self.k_cache[:batch_size,0:start_pos+seq_len]
        
        #broadcast through the entire head
        keys = repeat_kv(keys,self.r_rep)
        value = repeat_kv(value,self.r_rep)

        query_weight = query_weight.transpose(1,2)
        keys = key_weight.transpose(1,2)
        value = value_weight.transpose(1,2)

        #attention weight
        scores = torch.matmul(query_weight,keys.transpose(2,3))/math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(),dim=-1).type_as(query_weight)

        output = torch.matmul(scores,value)
        output = (output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))

        return self.wo(output)

def compute_theta_pos_freq(head_dim,seq_len,device,theta=1000.0):
    assert head_dim % 2 ==0
    theta_numerator = torch.arange(0,head_dim,2).float()
    theta = 1.0 / (theta ** (theta_numerator/head_dim)).to(device)

    m = torch.arange(seq_len,device=device)
    #multiply the positon m by the theta values
    freqs = torch.outer(m,theta).float()
    frequency_complex = torch.polar(torch.ones_like(freqs),freqs)
    return frequency_complex


def rotatry_postional_embedding(input,frequency_complex,device):
    # (B,T,nh,hs) -> (B,T,nh,hs/2)
    input_complex = torch.view_as_complex(input.float().reshape(*input.shape[-1],-1,2))
    # (T,hs/2) -> (1,T,1,hs/2)
    frequency_complex = frequency_complex.unsqueeze(0).unsqueeze(2)
    input_rotated = input_complex * frequency_complex
    out = torch.view_as_real(input_rotated)
    out = out.reshape(*input.shape)
    return out.type_as(input).to(device)


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.rmsnorm1 = RMSNorm(config.n_emb)
        self.sa = SelfAttention(config)
        self.swiglu = SwiGlu(config)
        self.rmsnorm2 = RMSNorm(config.n_emb,eps=config.norm_eps)
    def forward(self,x):
        x = x + self.rmsnorm1(self.sa(x))
        x = x + self.rmsnorm2(self.swiglu(x))
        
class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones_like(dim))
    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        return self.gamma * self._norm(x.float()).type_as(x)
    
class SwiGlu(nn.Module):
    def __init__(self,config):
        super().__init__()
        hidden_dim = 4 * config.n_emb
        hidden_dim = int(2 * hidden_dim / 3)
        if config.multiple_of is not None:
            hidden_dim = config.multiple_of *((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.n_emb, hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_emb, bias=False)
        self.w3 = nn.Linear(config.n_emb, hidden_dim, bias=False)
    def forward(self,x):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)
        return x
class LLamaTransformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        assert config.vocab_size != -1
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(self.vocab_size,config.n_emb)
        self.n_layers = config.n_layers
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.n_emb,eps=config.norm_eps)
        self.output = nn.Linear(config.n_emb,self.vocab_size,bias=False)
        self.frequency_complex = compute_theta_pos_freq(self.config.n_emb//self.config.n_heads,self.config.max_seq_len * 2,device=self.config.device)

    def forward(self,x,start_pos):
        batch_size,seq_len = x.shape # B,T
        assert seq_len == 1
        h = self.embeddings(x) # B,T -> B,T,C
        freq_complex = self.frequency_complex[start_pos:start_pos + seq_len]
        for layer in self.layers:
            h = layer(h,start_pos,freq_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output