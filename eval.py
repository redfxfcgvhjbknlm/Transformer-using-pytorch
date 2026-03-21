from nltk.translate.bleu_score import sentence_bleu

def compute_metrics(preds, targets):
    bleu_scores = []
    exact = 0

    for p, t in zip(preds, targets):
        bleu_scores.append(sentence_bleu([t.split()], p.split()))
        if p.strip() == t.strip():
            exact += 1

    return {
        "BLEU": sum(bleu_scores)/len(bleu_scores),
        "Exact Match": exact / len(preds)
    }
