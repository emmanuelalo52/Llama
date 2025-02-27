import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math
import torch.nn as nn
from typing import Optional

@dataclass
class LlamaConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def repeat_kv(x, n_rep):
    batch_size,seq_len,n_kv_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (x[:,:,:,None,:].expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim).reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim))

def lambda_init(depth):
    return (0.8 - 0.6 * math.exp(-0.3*(depth))) #depth is layer index
class SelfAttention(nn.Module):
    def __init__(self, config,depth):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = config.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = config.dim // config.n_heads

        self.lambda_init = lambda_init(depth)
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim, dtype=torch.float32) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim, dtype=torch.float32) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim, dtype=torch.float32) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim, dtype=torch.float32) * 0.1)
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x,
        start_pos,
        freqs_complex,

    ):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(xk, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(xv, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        #use diffrerential attention in the multi head attention
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # Modify the attention scores using lambda parameters
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).type_as(xq)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).type_as(xq)
        sum_lambda = lambda_1 - lambda_2 + self.lambda_init

        # Apply differential attention (ensure reshaping is correct)
        scores = scores.view(batch_size, self.n_heads_q, seq_len, seq_len)
        scores = scores - sum_lambda.unsqueeze(-1) * scores

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))    
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta  = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(input, frequency_complex, device):
    # Reshape the input to separate the real and imaginary parts
    input_reshaped = input.float().reshape(*input.shape[:-1], -1, 2)
    input_complex = torch.view_as_complex(input_reshaped)
    
    # Apply rotary embeddings
    frequency_complex = frequency_complex.unsqueeze(0).unsqueeze(2)
    input_rotated = input_complex * frequency_complex
    
    # Convert back to real numbers
    out = torch.view_as_real(input_rotated)
    out = out.reshape(*input.shape)
    return out.type_as(input).to(device)


class EncoderBlock(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.attention = SelfAttention(config, depth)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
    def forward(self, x, start_pos, freqs_complex):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
        
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x).type_as(x)
    

class FeedForward(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()

        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x
class LLamaTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.tok_embeddings = nn.Embedding(self.vocab_size, config.dim).to(self.device)
        self.layers = nn.ModuleList([EncoderBlock(config, depth=i).to(self.device) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps).to(self.device)
        self.output = nn.Linear(config.dim, self.vocab_size, bias=False).to(self.device)
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.config.dim // self.config.n_heads, self.config.max_seq_len * 2, device=self.device
        )

    def forward(self, x, start_pos):
        batch_size, seq_len = x.shape
        h = self.tok_embeddings(x)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output