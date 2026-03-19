class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ff    = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sub-layer 1: self-attention  (pre-norm variant)
        x = x + self.drop(self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        ))
        # Sub-layer 2: feed-forward
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, N, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(N)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x   # (B, src_len, d_model)
