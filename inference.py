import torch
from transformers import AutoTokenizer
from model import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("t5-small")

model = Transformer(...).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

def generate(src_text, max_len=100, beam_size=3):
    src = tokenizer(src_text, return_tensors="pt").input_ids.to(device)

    beams = [(src, 0)]

    for _ in range(max_len):
        new_beams = []

        for seq, score in beams:
            tgt = seq

            logits = model(src, tgt)
            probs = torch.log_softmax(logits[:, -1], dim=-1)

            topk = torch.topk(probs, beam_size)

            for i in range(beam_size):
                token = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([tgt, token], dim=1)
                new_score = score + topk.values[0, i].item()
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    return tokenizer.decode(beams[0][0][0])
