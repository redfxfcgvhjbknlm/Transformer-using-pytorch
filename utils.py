import torch

def create_masks(src, tgt, pad):
    src_mask = (src != pad).unsqueeze(1).unsqueeze(2)

    tgt_mask = (tgt != pad).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    causal = torch.tril(torch.ones(seq_len, seq_len)).bool().to(tgt.device)

    tgt_mask = tgt_mask & causal
    return src_mask, tgt_mask


def lr_schedule(step, d_model, warmup=4000):
    return d_model**(-0.5) * min(step**(-0.5), step * warmup**(-1.5))
