from torch import nn
import torch  

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None=None, 
                 dtype: torch.dtype | None=None):
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        self.device = device 
        self.dtype = dtype 

        self.W = nn.Parameter(torch.empty((self.in_features, self.out_features), device=self.device, dtype=self.dtype))
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean = 0, std = std, a = -3 * std, b = 3 * std) 

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x @ self.W 


class EmbeddingLayer(nn.Module):
    def __init__(self, 
                num_embeddings: int,
                embedding_dim: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim 
        self.device = device 
        self.dtype = dtype 

        self.embedding_matrix = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device = self.device, dtype = self.dtype)) 
        torch.nn.init.trunc_normal_(self.embedding_matrix, mean = 0, std = 1, a= -3, b = 3) 

    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, 
                d_model: int, 
                eps: float = 1e-5, 
                device=None, 
                dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        self.device = device 
        self.dtype = dtype 

        self.gamma = nn.Parameter(torch.ones(self.d_model, device = self.device, dtype = self.dtype)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x.to(torch.float32) 
        rms_mean = torch.sqrt(torch.square(input).mean(dim = -1, keepdim = True) + self.eps)
        out = input / rms_mean * self.gamma 
        return out.to(x.dtype) 


class SwiGlu(nn.Module):
    def __init__(self, 
                d_model: int, 
                d_ff: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model 
        self.d_ff = d_ff 
        self.device = device 
        self.dtype = dtype 

        self.W1 = Linear(self.d_model, self.d_ff, self.device, self.dtype) 
        self.W2 = Linear(self.d_ff, self.d_model, self.device, self.dtype)
        self.W3 = Linear(self.d_model, self.d_ff, self.device, self.dtype)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.W1(x)
        silu = w1_out / ( 1 + torch.exp(- w1_out)) 
        swi_glu = silu * self.W3(x) 
        return self.W2(swi_glu) 

class RoPE(nn.Module):
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None =None):
        super().__init__()
        self.theta = theta 
        self.d_k = d_k 
        self.max_seq_len = max_seq_len 
        self.device = device 

        pos = torch.arange(0, self.max_seq_len)
        d = d_k / 2 
        freq = self.theta ** ( - torch.arange(0, d) / d) 
        rote_matrix = torch.outer(pos, freq)
        self.register_buffer("cos_matrix", torch.cos(rote_matrix), persistent=False)
        self.register_buffer("sin_matrix", torch.sin(rote_matrix), persistent=False) 
        
    def forward(self, 
                x: torch.Tensor, 
                token_positions: torch.Tensor) -> torch.Tensor:
        x_0 = x[..., 0::2]
        x_1 = x[..., 1::2]
        cos_matrix = self.cos_matrix[token_positions]
        sin_matrix = self.sin_matrix[token_positions]
        print(f"cos_matirx shape is {cos_matrix.shape}, x_0 matrix shape is {x_0.shape}")

        output_0 = x_0 * cos_matrix - x_1 * sin_matrix 
        output_1 = x_0 * sin_matrix + x_1 * cos_matrix 
        return torch.stack([output_0, output_1], dim=-1).flatten(-2)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    dim_max = torch.max(x, dim = dim, keepdim=True)[0]
    x_exp = torch.exp(x - dim_max)
    norm = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / norm


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                querys: torch.Tensor, 
                keys: torch.Tensor, 
                values: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        Q = querys
        K = keys.transpose(-2, -1)
        d_k = querys.shape[-1]
        sdpa_score = torch.matmul(Q, K) / (d_k ** 0.5) 
        if mask is not None:
            sdpa_score = sdpa_score.masked_fill(mask == False, -1e9)
        V = values 
        return torch.matmul(softmax(sdpa_score), values) 


class MultiheadSelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device 
        self.dtype = dtype 

        self.Wq = Linear(self.d_model, self.d_model, self.device, self.device)
        self.Wk = Linear(self.d_model, self.d_model, self.device, self.device)
        self.Wv = Linear(self.d_model, self.d_model, self.device, self.device)
        self.Wo = Linear(self.d_model, self.d_model, self.device, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        q_proj = self.Wq(x) 
        k_proj = self.Wk(x)
        v_proj = self.Wv(x)
        q_heads = q_proj.view(-1, seq_len, self.num_heads, self.d_model // self.num_heads) #注意这里的除号，以及维度推断只能有一个-1
        k_heads = k_proj.view(-1, seq_len, self.num_heads, self.d_model // self.num_heads)
        v_heads = v_proj.view(-1, seq_len, self.num_heads, self.d_model // self.num_heads)

        q_heads = torch.permute(q_heads, (0, 2, 1, 3))
        k_heads = torch.permute(k_heads, (0, 2, 1, 3))
        v_heads = torch.permute(v_heads, (0, 2, 1, 3)) 

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype = torch.bool)) 

        sdpa = ScaledDotProductAttention()
        o_heads = sdpa(q_heads, k_heads, v_heads, mask) 
        o_heads = torch.permute(o_heads, (0, 2, 1, 3)).contiguous() 
        o = o_heads.view(-1, seq_len, self.d_model) #注意这里要写d_model，而不是d_model * num_heads
        output = self.Wo(o)
        return output 


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 max_seq_len: int,
                 theta: float, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads 
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device 
        self.dtype = dtype  

        self.Wq = Linear(self.d_model, self.d_model, self.device, self.device)
        self.Wk = Linear(self.d_model, self.d_model, self.device, self.device)
        self.Wv = Linear(self.d_model, self.d_model, self.device, self.device)
        self.Wo = Linear(self.d_model, self.d_model, self.device, self.device)
        self.rope = RoPE(theta, self.d_model // self.num_heads, max_seq_len, device=self.device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        q_proj = self.Wq(x) 
        k_proj = self.Wk(x)
        v_proj = self.Wv(x)

        q_heads = q_proj.view(-1, seq_len, self.num_heads, self.d_model // self.num_heads) #注意这里的除号，以及维度推断只能有一个-1
        k_heads = k_proj.view(-1, seq_len, self.num_heads, self.d_model // self.num_heads)
        v_heads = v_proj.view(-1, seq_len, self.num_heads, self.d_model // self.num_heads)

        q_heads = torch.permute(q_heads, (0, 2, 1, 3))
        k_heads = torch.permute(k_heads, (0, 2, 1, 3))
        v_heads = torch.permute(v_heads, (0, 2, 1, 3)) 

        if token_positions is not None:
            rope_heads = token_positions.unsqueeze(dim = 1).expand(q_heads.shape[0], self.num_heads, seq_len)
            q_heads = self.rope(q_heads, rope_heads)
            k_heads = self.rope(k_heads, rope_heads)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype = torch.bool)) 

        sdpa = ScaledDotProductAttention()
        o_heads = sdpa(q_heads, k_heads, v_heads, mask) 
        o_heads = torch.permute(o_heads, (0, 2, 1, 3)).contiguous() 
        o = o_heads.view(-1, seq_len, self.d_model) #注意这里要写d_model，而不是d_model * num_heads
        output = self.Wo(o)
        return output 

class Transformer_Block(nn.Module):
    def __init__(self, 
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_ff = d_ff 
        self.max_seq_len = max_seq_len 
        self.theta = theta
        self.device = device 
        self.dtype = dtype 

        self.rms_norm1 = RMSNorm(self.d_model, 0.00001, self.device, self.dtype) 
        self.mhsarope = MultiheadSelfAttentionRoPE(self.d_model, self.num_heads, self.max_seq_len, self.theta, self.device, self.dtype)
        self.rms_norm2 = RMSNorm(self.d_model, 0.00001, self.device, self.dtype)
        self.ffn = SwiGlu(self.d_model, self.d_ff, self.device, self.dtype) 

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        seq_len = x.shape[-2] 
        batch_size = x.shape[-3]
        token_pos = torch.unsqueeze(torch.arange(seq_len), dim = 0).expand([batch_size, seq_len])
        x_norm1 = self.rms_norm1(x) 
        x_mhsarope = self.mhsarope(x_norm1, token_pos) 
        x_o1 = x_mhsarope + x
        x_norm2 = self.rms_norm2(x_o1) 
        x_ffn = self.ffn(x_norm2) 
        x_o2 = x_ffn + x_o1
        return x_o2 


class TransformerLm(nn.Module):
    def __init__(self, 
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float, 
                device: torch.device | None = None,
                dtype: torch.dtype | None = None
                ):
        super().__init__()
        self.vocab_size = vocab_size 
        self.context_length = context_length 
        self.d_model = d_model 
        self.num_layers = num_layers 
        self.num_heads = num_heads 
        self.d_ff = d_ff 
        self.rope_theta = rope_theta 
        self.device = device 
        self.dtype = dtype 

        self.token_embeddings = EmbeddingLayer(self.vocab_size, self.d_model, self.device, self.dtype) 
        self.layers = nn.ModuleList(
            [Transformer_Block(self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta, self.device, self.dtype) for i in range(self.num_layers)]
        )
        # for i in range(self.num_layers):
        #     setattr(self, f'layers.{i}', Transformer_Block(self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta, self.device, self.dtype))
        self.ln_final = RMSNorm(d_model, 0.00001, self.device, self.dtype) 
        self.lm_head = Linear(self.d_model, self.vocab_size, self.device, self.dtype) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x) 
        # for i in range(self.num_layers):
        #     layer = getattr(self, f'layers.{i}')
        #     x = layer(x) 
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x) 
        x = self.lm_head(x) 
        return x 
        
