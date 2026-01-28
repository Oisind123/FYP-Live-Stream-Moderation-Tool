import pytchat
from transformers import pipeline
import torch

MODEL_NAME = "unitary/toxic-bert"
device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device)

chat = pytchat.create(video_id="rUO61GYpLPU")
THRESH = 0.60
THRESH1 = 0.20

def get_scores(text: str):
    res = clf(text, truncation=True, return_all_scores=True)

    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        items = res[0]
    elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        items = res
    else:
        return {}

    return {d["label"].upper(): float(d["score"]) for d in items}

while chat.is_alive():
    for c in chat.get().sync_items():
        text = c.message
        scores = get_scores(text)

        p_toxic = scores.get("TOXIC", scores.get("LABEL_1", 0.0))

        is_likelyToxic = p_toxic >= THRESH
        is_ToxicElements = THRESH1 <= p_toxic < THRESH

        if is_likelyToxic:
            tag = "Likely Toxic Chat"
        elif is_ToxicElements:
            tag = "Toxic Elements in Chat"
        else:
            tag = "Normal Chat"

        print(f"THRESH={THRESH} | {c.datetime} [{c.author.name}] {tag} (TOXIC:{p_toxic:.2f}) - {text}")