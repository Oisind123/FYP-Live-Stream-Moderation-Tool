import pytchat
from transformers import pipeline
import torch
import re

MODEL_NAME = "unitary/toxic-bert"
device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device)

# -------- Stream input --------
print("\n=== YouTube Live Toxicity Monitor ===")
print("Insert YouTube live stream link here\n")

user_input = input("Stream link: ").strip()

def extract_video_id(s: str):
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", s):
        return s

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"live/([a-zA-Z0-9_-]{11})",
    ]

    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)

    return None

video_id = extract_video_id(user_input)

if not video_id:
    print("Could not extract video ID from input.")
    print("Example formats:")
    print("https://www.youtube.com/watch?v=dxoP2P5l8J8")
    print("https://youtu.be/dxoP2P5l8J8")
    print("dxoP2P5l8J8")
    exit(1)

print(f"\nConnected to stream: {video_id}\n")

# chat model
chat = pytchat.create(video_id=video_id)

THRESH  = 0.60
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

print("Monitoring chat...\n")

while chat.is_alive():
    for c in chat.get().sync_items():
        text = c.message
        scores = get_scores(text)

        p_toxic = scores.get("TOXIC", scores.get("LABEL_1", 0.0))

        is_likelyToxic  = p_toxic >= THRESH
        is_ToxicElements = THRESH1 <= p_toxic < THRESH

        if is_likelyToxic:
            tag = "Likely Toxic Chat"
        elif is_ToxicElements:
            tag = "Toxic Elements in Chat"
        else:
            tag = "Normal Chat"

        print(f"{c.datetime} [{c.author.name}] {tag} (TOXIC:{p_toxic:.2f}) - {text}")