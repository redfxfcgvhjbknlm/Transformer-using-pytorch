class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab, tgt_vocab,
        d_model=512, num_heads=8,
        N=6, d_ff=2048,
        max_len=5000, dropout=0.1
    ):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.encoder   = Encoder(d_model, num_heads, d_ff, N, dropout)
        self.decoder   = Decoder(d_model, num_heads, d_ff, N, dropout)
        self.fc_out    = nn.Linear(d_model, tgt_vocab)
        self.scale     = math.sqrt(d_model)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for linear layers — important for training stability
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=0):
        # (B, 1, 1, src_len) — broadcast over heads
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, L = tgt.shape
        # pad mask ∩ causal mask
        pad_mask    = (tgt != 0).unsqueeze(1).unsqueeze(3)
        causal_mask = torch.tril(torch.ones(L, L, device=tgt.device)).bool()
        return pad_mask & causal_mask   # (B, 1, tgt_len, tgt_len)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        src_emb  = self.pos_enc(self.src_embed(src) * self.scale)
        tgt_emb  = self.pos_enc(self.tgt_embed(tgt) * self.scale)

        enc_out  = self.encoder(src_emb, src_mask)
        dec_out  = self.decoder(tgt_emb, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)   # (B, tgt_len, tgt_vocab)
