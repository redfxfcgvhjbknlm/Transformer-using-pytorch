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


# Training Loop

criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD tokens
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4, betas=(0.9, 0.98), eps=1e-9
)

for epoch in range(num_epochs):
    model.train()
    for src, tgt in dataloader:
        # Teacher forcing: feed tgt[:-1], predict tgt[1:]
        logits = model(src, tgt[:, :-1])  # (B, tgt_len-1, vocab)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
