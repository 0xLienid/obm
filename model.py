import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional
from model_utils import gumbel_softmax

import torch
import torch.nn.functional as F
import torch.distributions as dist
from torch import nn


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_regions: int = 8
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class RegularizationLoss(nn.Module):
    def __init__(self, lambda_p: float, max_steps: int):
        super().__init__()
        self.lambda_p = lambda_p
        self.max_steps = max_steps

        not_halted = 1.0
        p_g_list = []
        for _ in range(self.max_steps):
            p_g_list.append(not_halted * self.lambda_p)
            not_halted *= (1 - self.lambda_p)
        p_g = torch.tensor(p_g_list, dtype=torch.float32)
        self.register_buffer('p_g', p_g)

    def forward(self, p):
        p = p.transpose(1, 0)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p).to("cuda")
        result = F.kl_div(torch.log(p + 1e-9), p_g, reduction='batchmean')

        return result


class RegionRouter(nn.Module):
    def __init__(self, input_dim: int, n_regions: int, seq_len: int, max_regions: int):
        super().__init__()
        self.seq_len = seq_len
        self.n_regions = n_regions

        self.w1 = nn.Linear(input_dim + max_regions + 1, n_regions, bias=False)
        self.w2 = nn.Linear(seq_len * n_regions, n_regions, bias=False)
        self.norm = RMSNorm(n_regions, eps=1e-5)

    def forward(self, x, regions_used, last_regions):
        _, n, _ = x.shape
        if n < self.seq_len:
            pad_size = self.seq_len - n
            x = F.pad(x, (0, 0, 0, pad_size), "constant", 0)

        regions_used_tensor = torch.tensor(
            [[regions_used]], dtype=x.dtype, device=x.device).repeat(1, self.seq_len, 1)
        last_regions_tensor = torch.tensor(
            [last_regions], dtype=x.dtype, device=x.device).repeat(1, self.seq_len, 1)
        x = torch.cat((x, regions_used_tensor, last_regions_tensor), -1)

        x = self.w1(x)
        x = x.view(-1, self.seq_len * self.n_regions)
        x = self.w2(x)
        x = self.norm(x)
        return x


class HaltRouter(nn.Module):
    def __init__(self, input_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.w1 = nn.Linear(input_dim, 1, bias=False)
        self.w2 = nn.Linear(seq_len, 1, bias=False)

    def forward(self, x):
        _, n, _ = x.shape
        if n < self.seq_len:
            pad_size = self.seq_len - n
            x = F.pad(x, (0, 0, 0, pad_size), "constant", 0)

        x = self.w1(x)
        x = x.view(-1, self.seq_len)
        x = self.w2(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(args.dim, 3 * args.dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(args.dim, args.dim, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.dropout = args.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(args.max_seq_len, args.max_seq_len))
                                 .view(1, 1, args.max_seq_len, args.max_seq_len))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.dim, dim=2)
        k = k.view(B, T, self.n_heads, C //
                   self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C //
                   self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C //
                   self.n_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * \
                ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.depth_router = nn.Linear(args.dim, 1, bias=False)
        self.dr_norm = RMSNorm(1, eps=args.norm_eps)
        self.attention = CausalSelfAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x,
        capacity: float = 1.0,
        attn_mask: Optional[torch.Tensor] = None
    ):
        x_input = None
        indices_expanded = None
        weights = None

        if capacity == 1.0:
            x_input = x
        else:
            _, n, m = x.shape

            capacity = max(1, int(capacity * n))
            token_logits = self.dr_norm(self.depth_router(x))
            weights, selected_tokens = torch.topk(
                token_logits, capacity, dim=1, sorted=False)

            selected_tokens, index = torch.sort(selected_tokens, dim=1)
            weights = torch.gather(weights, dim=1, index=index)
            indices_expanded = selected_tokens.expand(-1, -1, m)
            x_input = torch.gather(x, 1, indices_expanded)

        h = x_input + \
            self.attention.forward(
                self.attention_norm(x_input))
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        if capacity != 1.0:
            x = torch.scatter_add(
                x,
                dim=1,
                index=indices_expanded,
                src=h * weights,
            )
            return x

        return out


class Region(nn.Module):
    def __init__(self, id: int, depth: int, params: ModelArgs):
        super().__init__()
        self.id = id
        self.global_id = id * depth
        self.depth = depth
        self.params = params
        self.blocks = torch.nn.ModuleList()
        for block_id in range(depth):
            self.blocks.append(TransformerBlock(
                self.global_id + block_id, params))

        # Initialize attribute for MoD capacity
        self.capacity = 0.125

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            block_capacity = 1.0 if i % 2 == 0 else self.capacity
            x = block(x, block_capacity)
        return x


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.n_regions = params.n_regions
        self.region_depth = params.n_layers // params.n_regions

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.pos_embeddings = nn.Embedding(params.max_seq_len, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.preprocessing_block = TransformerBlock(98, params)
        self.halt_router = HaltRouter(params.dim, params.max_seq_len)
        self.region_router = RegionRouter(
            params.dim, params.n_regions, params.max_seq_len, self.n_regions * 4)
        self.regions = torch.nn.ModuleList()
        for region_id in range(params.n_regions):
            self.regions.append(Region(region_id, self.region_depth, params))
        self.postprocessing_block = TransformerBlock(99, params)
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.reg_loss_module = RegularizationLoss(
            0.2, self.n_regions * 4)

        # share the unembedding parameters with the embedding parameters
        # https://paperswithcode.com/method/weight-tying
        self.tok_embeddings.weight = self.output.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attributes for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_base_loss = None
        self.last_total_loss = None
        self.last_regions_used = None
        self.region_usage = None

        # Initialize attributes for block usage
        # theoretically there should be no max block usage, but using this for now
        self.max_region_usage = self.n_regions * 4
        self.region_penalty = 0.5

        # Initialize attribute for MoD capacity
        self.capacity = 0.125

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_halt_router_param_count(self):
        return sum(p.numel() for p in self.halt_router.parameters())

    def get_region_router_param_count(self):
        return sum(p.numel() for p in self.region_router.parameters())

    def get_per_block_param_count(self):
        return sum(p.numel() for p in self.regions[0].parameters())

    # currently only works for batch size 1
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        pos_emb = self.pos_embeddings(torch.arange(
            0, seqlen, dtype=tokens.dtype, device=tokens.device).unsqueeze(0))
        h = self.dropout(h + pos_emb)

        h = self.preprocessing_block(h)

        unhalted_prob = 1.0
        p = []
        regions_used = 0
        last_regions = [-1] * self.max_region_usage

        while regions_used < self.max_region_usage:
            lambda_n = torch.sigmoid(self.halt_router(h))
            p_n = unhalted_prob * lambda_n
            unhalted_prob = unhalted_prob * (1 - lambda_n)
            halt = dist.Bernoulli(lambda_n - 1.1).sample().item()
            p.append(p_n)

            if halt == 1.0:
                break

            # get the region to route to by the max of the router
            logits = self.region_router(
                h, regions_used, last_regions)
            probabilities = gumbel_softmax(
                logits, temperature=0.1, hard=False)
            region = torch.multinomial(probabilities.squeeze(), 1).item()

            try:
                # Find the first occurrence of -1 and replace it
                index = last_regions.index(-1)
                last_regions[index] = region
            except ValueError:
                # No -1 found in the array, so remove the first number and append the new one
                last_regions.pop(0)
                last_regions.append(region)

            regions_used += 1
            h = self.regions[region](h)

        h = self.postprocessing_block(h)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)

            targets = targets.view(-1)
            attn_mask = attn_mask.view(-1)
            targets[attn_mask == 0] = -100

            self.last_base_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets, ignore_index=-100)
            self.last_total_loss = self.last_base_loss + \
                self.reg_loss_module(torch.tensor([p], device="cuda")) + self.region_penalty * (
                    ((regions_used - self.n_regions) / self.n_regions) ** 2)
            self.last_regions_used = regions_used
            self.region_usage = dict((region, self.last_regions_used.count(
                region)) for region in set(self.last_regions_used))
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.output(h[:, [-1], :])
            self.last_base_loss = None
            self.last_total_loss = None
            self.last_regions_used = None
            self.region_usage = None

        return logits

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                try:
                    idx_next = torch.multinomial(probs, num_samples=1)
                except Exception as e:
                    idx_next = torch.tensor([50256]).unsqueeze(0).to("cuda")
                    print(f"Error: {e}, probs tensor: {probs}")
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
