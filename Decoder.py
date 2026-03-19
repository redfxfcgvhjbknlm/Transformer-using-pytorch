class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff    = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention  (can't see future tokens)
        x = x + self.drop(self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask
        ))
        # 2. Cross-attention  (Q ← decoder, K/V ← encoder)
        x = x + self.drop(self.cross_attn(
            self.norm2(x), enc_out, enc_out, src_mask
        ))
        # 3. Feed-forward
        x = x + self.drop(self.ff(self.norm3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, N, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(N)
        ])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x   # (B, tgt_len, d_model)
