"""
model.py - Transformer Encoder for Waveform Selection.

Architecture:
  TokenEmbedding(57, 64) + LearnedPositionalEncoding(12, 64)
  → 2× TransformerEncoderLayer(d=64, h=4, ff=128, dropout=0.1)
  → GlobalMeanPooling → Linear(64,32) → ReLU → Dropout → Linear(32,6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# =========================================================================
# Configuration
# =========================================================================
VOCAB_SIZE = 57        # from tokenizer
SEQ_LEN = 12           # 12 domain tokens
EMBED_DIM = 64
N_HEADS = 4
HEAD_DIM = EMBED_DIM // N_HEADS  # 16
FF_DIM = 128
N_LAYERS = 2
DROPOUT = 0.1
N_CLASSES = 6          # 6 waveform candidates


# =========================================================================
# Learned Positional Encoding
# =========================================================================
class LearnedPositionalEncoding(nn.Module):
    """Learned position embedding for 12 semantic token positions.

    Unlike sinusoidal PE, this is learned because each position has a fixed
    semantic meaning (pos 0 = SNR, pos 9 = traffic type, etc.).
    """

    def __init__(self, max_len: int = SEQ_LEN, d_model: int = EMBED_DIM):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to input.

        Parameters
        ----------
        x : Tensor, shape (B, S, D)

        Returns
        -------
        Tensor, shape (B, S, D)
        """
        B, S, D = x.shape
        positions = torch.arange(S, device=x.device)  # (S,)
        return x + self.position_embedding(positions).unsqueeze(0)  # broadcast


# =========================================================================
# Multi-Head Self-Attention
# =========================================================================
class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention.

    Attention(Q,K,V) = softmax(Q K^T / √d_k) V
    4 heads, head_dim=16, total_dim=64.
    """

    def __init__(self, d_model: int = EMBED_DIM, n_heads: int = N_HEADS,
                 dropout: float = DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 16

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, S, D)

        Returns
        -------
        Tensor, shape (B, S, D)
        """
        B, S, D = x.shape
        h, d_k = self.n_heads, self.head_dim

        Q = self.W_Q(x).view(B, S, h, d_k).transpose(1, 2)  # (B, h, S, d_k)
        K = self.W_K(x).view(B, S, h, d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, h, d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, h, S, S)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, h, S, d_k)
        context = context.transpose(1, 2).contiguous().view(B, S, D)  # (B, S, D)

        return self.W_O(context)


# =========================================================================
# Transformer Encoder Layer
# =========================================================================
class TransformerEncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer.

    MHSA → Add & LayerNorm → FFN → Add & LayerNorm
    FFN = Linear(64,128) → ReLU → Linear(128,64)
    """

    def __init__(self, d_model: int = EMBED_DIM, ff_dim: int = FF_DIM,
                 n_heads: int = N_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual + norm (post-norm)
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout1(attn_out))

        # FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


# =========================================================================
# Full Waveform Transformer Encoder
# =========================================================================
class WaveformTransformerEncoder(nn.Module):
    """Complete waveform selection transformer.

    TokenEmbedding + LearnedPE → N encoder layers → mean pool → classifier.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int = SEQ_LEN,
        d_model: int = EMBED_DIM,
        n_heads: int = N_HEADS,
        ff_dim: int = FF_DIM,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = LearnedPositionalEncoding(seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, ff_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        token_ids : LongTensor, shape (B, 12)

        Returns
        -------
        logits : Tensor, shape (B, 6)
        """
        x = self.token_embedding(token_ids)            # (B, 12, 64)
        x = self.positional_encoding(x)                # (B, 12, 64)
        x = self.embed_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)                               # (B, 12, 64)

        # Global mean pooling over sequence dimension
        x = x.mean(dim=1)                              # (B, 64)

        logits = self.classifier(x)                    # (B, 6)
        return logits

    def predict(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return argmax predicted class.

        Returns
        -------
        Tensor, shape (B,) — integer class indices.
        """
        logits = self.forward(token_ids)
        return logits.argmax(dim=-1)

    def predict_with_confidence(
        self, token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predicted class, confidence, and full probability distribution.

        Returns
        -------
        predicted_class : Tensor, shape (B,)
        confidence : Tensor, shape (B,)
        probabilities : Tensor, shape (B, 6)
        """
        logits = self.forward(token_ids)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        return predicted, confidence, probs


# =========================================================================
# Utility
# =========================================================================
def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
