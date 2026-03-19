model = Transformer(
    src_vocab=10_000,
    tgt_vocab=10_000,
    d_model=512,
    num_heads=8,
    N=6,
    d_ff=2048,
    dropout=0.1,
)

src = torch.randint(1, 10_000, (4, 32))  # batch=4, src_len=32
tgt = torch.randint(1, 10_000, (4, 28))  # batch=4, tgt_len=28

out = model(src, tgt)
print(out.shape)  # → torch.Size([4, 28, 10000])

# Parameter count
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")  # → ~44M for these settings
