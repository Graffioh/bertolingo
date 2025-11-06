"""
Model definitions for the translation transformer
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Embedding layer for the tokens
    
    Args:
        vocab_size: size of the vocabulary
        d_embd: dimension of the embeddings
        padding_idx: index of the padding token
    """
    
    def __init__(self, vocab_size, d_embd, padding_idx=None):
        super().__init__()
        # for translation task we should have padding_idx = stoi['<PAD>'] to identify the padding tokens
        self.embd = nn.Embedding(vocab_size, d_embd, padding_idx=padding_idx)
    
    def forward(self, x):
        return self.embd(x)


class PositionalEmbedding(nn.Module):
    """
    Embedding layer for the positional encodings
    
    Args:
        n_tokens: number of tokens in the sequence
        d_embd: dimension of the embeddings
    """
    
    def __init__(self, n_tokens, d_embd):
        super().__init__()
        self.embd = nn.Embedding(n_tokens, d_embd)
    
    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return self.embd(pos)


class Head(nn.Module):
    """
    Single Head of the attention mechanism
    
    Args:
        d_embd: dimension of the embeddings
        head_size: dimension of the head
        dropout: dropout rate
        context_window: maximum context window size
    """
    
    def __init__(self, d_embd, head_size, dropout=0.1, context_window=128):
        super().__init__()
        self.query = nn.Linear(d_embd, head_size, bias=False)
        self.key = nn.Linear(d_embd, head_size, bias=False)
        self.value = nn.Linear(d_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))
    
    def forward(self, x, src_kv=None, key_padding_mask=None, causal_mask=False):
        """
        Args:
            x: (B, T, d_embd) - Input tensor
            src_kv: (B, T, d_embd) - Source key and value tensor
            key_padding_mask: (B, T) - Boolean mask (True for padding positions)
            causal_mask: bool - Whether to apply causal mask
        """
        _, q_pos, _ = x.shape
        
        # (B, T, d_embd) -> (B, pos, head_size)
        q = self.query(x)
        if src_kv is not None:
            k = self.key(src_kv)
            v = self.value(src_kv)
        else:
            k = self.key(x)
            v = self.value(x)
        
        # (B, q_pos, head_size) @ (B, (k_pos, head_size)^T) -> (B, q_pos, k_pos)
        qk = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        
        # for decoder
        if causal_mask:
            # Note: k_pos = q_pos
            qk = qk.masked_fill(self.tril[:q_pos, :q_pos] == 0, float('-inf'))  # (B, q_pos, q_pos)
        
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1)  # (B, 1, k_pos)
            qk = qk.masked_fill(expanded_mask, float('-inf'))  # (B, q_pos, k_pos)
        
        attn = torch.softmax(qk, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of attention in parallel
    
    Args:
        d_embd: dimension of the embeddings
        n_heads: number of attention heads
        dropout: dropout rate
        context_window: maximum context window size
    """
    
    def __init__(self, d_embd, n_heads, dropout=0.1, context_window=128):
        super().__init__()
        assert d_embd % n_heads == 0, "d_embd must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.head_size = d_embd // n_heads
        
        # Create multiple heads in parallel
        self.heads = nn.ModuleList([
            Head(d_embd, self.head_size, dropout, context_window) 
            for _ in range(n_heads)
        ])
        
        self.proj = nn.Linear(d_embd, d_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src=None, key_padding_mask=None, causal_mask=False):
        """
        Args:
            x: (B, T_q, d_embd) - Query input tensor
            src: (B, T_kv, d_embd) - Key/Value input tensor (for cross-attention)
            key_padding_mask: (B, T_kv) - Boolean mask (True for padding positions)
            causal_mask: bool - Whether to apply causal mask
        Returns:
            (B, T_q, d_embd) - Attention output
        """
        # Run all heads in parallel and concatenate outputs
        # Each head outputs (B, T_q, head_size)
        out = torch.cat([
            h(x, src, key_padding_mask, causal_mask) for h in self.heads
        ], dim=-1)
        
        # Output projection and dropout
        out = self.dropout(self.proj(out))
        
        return out


class MLP(nn.Module):
    """
    Multi-layer perceptron
    
    Args:
        d_embd: dimension of the embeddings
        dropout: dropout rate
    """
    
    def __init__(self, d_embd, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embd)
        self.mlp = nn.Sequential(
            nn.Linear(d_embd, 4 * d_embd),
            nn.GELU(),
            nn.Linear(4 * d_embd, d_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.mlp(self.ln1(x))


class EncoderBlock(nn.Module):
    """
    Encoder block with:
    1. Multi-Head Self-attention
    2. Feed-forward network
    
    Args:
        d_embd: dimension of the embeddings
        n_heads: number of attention heads
        dropout: dropout rate
        context_window: maximum context window size
    """
    
    def __init__(self, d_embd, n_heads, dropout=0.1, context_window=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embd)
        self.attn = MultiHeadAttention(d_embd, n_heads, dropout, context_window)
        self.ln2 = nn.LayerNorm(d_embd)
        self.mlp = MLP(d_embd, dropout)
    
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (B, T, d_embd) - Input tensor
            key_padding_mask: (B, T) - Boolean mask (True for padding positions)
        Returns:
            (B, T, d_embd) - Output tensor
        """
        x = x + self.attn(self.ln1(x), src=None, key_padding_mask=key_padding_mask, causal_mask=False)
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    """
    Encoder with multiple blocks of multi-head self-attention and feed-forward networks
    
    Args:
        vocab_size: size of the vocabulary
        d_embd: dimension of the embeddings
        n_heads: number of attention heads
        dropout: dropout rate
        n_blocks: number of blocks in the encoder
        context_window: maximum context window size
        padding_idx: index of padding token
    """
    
    def __init__(self, vocab_size, d_embd, n_heads, dropout=0.1, n_blocks=4, 
                 context_window=128, padding_idx=None):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_embd, padding_idx=padding_idx)
        self.pos_emb = PositionalEmbedding(context_window, d_embd)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_embd, n_heads, dropout, context_window) 
            for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_embd)
    
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (B, T) - Input token indices
            key_padding_mask: (B, T) - Boolean mask (True for padding positions)
        Returns:
            (B, T, d_embd) - Encoded representations
        """
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        
        x = self.ln_f(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with:
    1. Masked multi-head self-attention (causal)
    2. Multi-head cross-attention to encoder output
    3. Feed-forward network
    
    Args:
        d_embd: dimension of the embeddings
        n_heads: number of attention heads
        dropout: dropout rate
        context_window: maximum context window size
    """
    
    def __init__(self, d_embd, n_heads, dropout=0.1, context_window=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embd)
        self.ln2 = nn.LayerNorm(d_embd)
        self.ln3 = nn.LayerNorm(d_embd)
        
        self.self_attn = MultiHeadAttention(d_embd, n_heads, dropout, context_window)
        self.cross_attn = MultiHeadAttention(d_embd, n_heads, dropout, context_window)
        
        self.mlp = MLP(d_embd, dropout)
    
    def forward(self, x, encoder_out, tgt_key_padding_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: (B, T_tgt, d_embd) - Decoder input
            encoder_out: (B, T_src, d_embd) - Encoder output
            tgt_key_padding_mask: (B, T_tgt) - Decoder padding mask
            src_key_padding_mask: (B, T_src) - Encoder padding mask
        """
        # 1. Masked self-attention (decoder attends to previous positions)
        x = x + self.self_attn(
            self.ln1(x),
            src=None,
            key_padding_mask=tgt_key_padding_mask,
            causal_mask=True
        )
        
        # 2. Cross-attention (decoder attends to encoder output)
        x = x + self.cross_attn(
            self.ln2(x),
            src=encoder_out,
            key_padding_mask=src_key_padding_mask,
            causal_mask=False
        )
        
        x = x + self.mlp(self.ln3(x))
        
        return x


class Decoder(nn.Module):
    """
    Decoder with multiple blocks of masked multi-head self-attention and cross-attention
    
    Args:
        vocab_size: size of the vocabulary
        d_embd: dimension of the embeddings
        n_heads: number of attention heads
        dropout: dropout rate
        n_blocks: number of decoder blocks
        context_window: maximum context window size
        padding_idx: index of padding token
    """
    
    def __init__(self, vocab_size, d_embd, n_heads, dropout=0.1, n_blocks=4,
                 context_window=128, padding_idx=None):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_embd, padding_idx=padding_idx)
        self.pos_emb = PositionalEmbedding(context_window, d_embd)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_embd, n_heads, dropout, context_window) 
            for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_embd)
        self.lm_head = nn.Linear(d_embd, vocab_size)
    
    def forward(self, x, encoder_out, tgt_key_padding_mask=None, src_key_padding_mask=None):
        """
        Returns:
            (B, T_tgt, vocab_size) - Logits for each position
        """
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x, encoder_out, tgt_key_padding_mask, src_key_padding_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


class Seq2SeqModel(nn.Module):
    """
    Complete Seq2Seq Translation Model combining Encoder and Decoder
    """
    
    def __init__(self, vocab_size, d_embd, n_heads, dropout=0.1, 
                 n_encoder_blocks=4, n_decoder_blocks=4, context_window=128, padding_idx=None):
        super().__init__()
        self.encoder = Encoder(
            vocab_size, d_embd, n_heads, dropout, n_encoder_blocks, 
            context_window, padding_idx
        )
        self.decoder = Decoder(
            vocab_size, d_embd, n_heads, dropout, n_decoder_blocks,
            context_window, padding_idx
        )
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            src: (B, T_src) - Source token indices (English)
            tgt: (B, T_tgt) - Target token indices (Italian)
            src_key_padding_mask: (B, T_src) - Source padding mask
            tgt_key_padding_mask: (B, T_tgt) - Target padding mask
        Returns:
            logits: (B, T_tgt, vocab_size) - Logits for each target position
        """
        # Encode source
        encoder_out = self.encoder(src, key_padding_mask=src_key_padding_mask)
        
        # Decode (with teacher forcing during training)
        logits = self.decoder(
            tgt,
            encoder_out,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return logits

