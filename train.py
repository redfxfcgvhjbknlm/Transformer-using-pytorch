import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import load_data, CodeDataset, tokenizer
from transformer.py import Transformer
from utils import create_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = CodeDataset(load_data("train"))
loader = DataLoader(train_data, batch_size=8, shuffle=True)

model = Transformer(
    src_vocab=tokenizer.vocab_size,
    tgt_vocab=tokenizer.vocab_size,
    d_model=512,
    num_heads=8,
    N=6,
    d_ff=2048,
    dropout=0.1,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask = create_masks(src, tgt_in, tokenizer.pad_token_id)

        logits = model(src, tgt_in, src_mask, tgt_mask)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")
