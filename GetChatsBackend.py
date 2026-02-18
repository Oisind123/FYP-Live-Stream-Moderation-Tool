import asyncio
import re
import threading
import time
import json
from typing import Optional, Set

import pytchat
import torch
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

MODEL_NAME = "unitary/toxic-bert"
THRESH_LIKELY = 0.60
THRESH_ELEMENTS = 0.20

app = FastAPI(title="Live Stream Moderation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/dashboard.html")


device = 0 if torch.cuda.is_available() else -1
clf = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=device
)


def get_scores(text: str) -> dict:
    res = clf(text, truncation=True, return_all_scores=True)

    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        items = res[0]
    elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        items = res
    else:
        return {}

    return {d["label"].upper(): float(d["score"]) for d in items}


clients: Set["asyncio.Queue[str]"] = set()
event_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_json(payload: str):
    if event_loop is None:
        print("BROADCAST DROPPED: event_loop is None", flush=True)
        return

    if len(clients) == 0:
        print("BROADCAST DROPPED: no clients", flush=True)
        return

    async def _push():
        print("PUSHING TO", len(clients), "clients", flush=True)
        for q in list(clients):
            try:
                q.put_nowait(payload)
            except Exception as e:
                print("QUEUE PUSH ERROR", repr(e), flush=True)

    asyncio.run_coroutine_threadsafe(_push(), event_loop)


worker_thread: Optional[threading.Thread] = None
stop_flag = threading.Event()
current_video_id: Optional[str] = None


def extract_video_id(s: str) -> Optional[str]:
    s = s.strip()

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


def tier_for_score(p_toxic: float) -> str:
    if p_toxic >= THRESH_LIKELY:
        return "LIKELY_TOXIC"
    if p_toxic >= THRESH_ELEMENTS:
        return "TOXIC_ELEMENTS"
    return "NORMAL"


def chat_worker(video_id: str):
    global current_video_id
    current_video_id = video_id

    chat = pytchat.create(video_id=video_id, interruptable=False)

    while not stop_flag.is_set() and chat.is_alive():
        try:
            for c in chat.get().sync_items():
                if stop_flag.is_set():
                    break

                text = c.message
                scores = get_scores(text)
                p_toxic = scores.get("TOXIC", scores.get("LABEL_1", 0.0))
                tier = tier_for_score(p_toxic)

                open_chat = f"https://www.youtube.com/live_chat?v={video_id}"
                open_watch = f"https://www.youtube.com/watch?v={video_id}"
                author_q = c.author.name.strip()
                search_user = f"https://www.youtube.com/results?search_query={author_q}"

                print("CHAT:", text, p_toxic, tier)

                payload_obj = {
                    "ts": str(c.datetime),
                    "video_id": video_id,
                    "author": author_q,
                    "text": text,
                    "p_toxic": float(p_toxic),
                    "tier": tier,
                    "links": {
                        "open_chat": open_chat,
                        "open_watch": open_watch,
                        "search_user": search_user,
                    },
                }

                print("BROADCASTING:", payload_obj)
                broadcast_json(json.dumps(payload_obj, ensure_ascii=False))

        except Exception as e:
            err_payload = (
                '{'
                f'"ts":"",'
                f'"video_id":"{video_id}",'
                f'"author":"",'
                f'"text":"[Backend error] {escape_json(str(e))}",'
                f'"p_toxic":0.0,'
                f'"tier":"SYSTEM",'
                f'"links":{{"open_chat":"","open_watch":"","search_user":""}}'
                '}'
            )
            broadcast_json(err_payload)
            time.sleep(1)

    current_video_id = None


def escape_json(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\r", "\\r")
         .replace("\t", "\\t")
    )


class StartReq(BaseModel):
    stream: str


@app.post("/start")
def start(req: StartReq):
    global worker_thread

    vid = extract_video_id(req.stream)
    if not vid:
        return {"ok": False, "error": "Could not extract video ID from input."}

    stop()

    stop_flag.clear()
    worker_thread = threading.Thread(
        target=chat_worker,
        args=(vid,),
        daemon=True
    )
    worker_thread.start()

    return {"ok": True, "video_id": vid}


@app.post("/stop")
def stop():
    global worker_thread

    stop_flag.set()

    if worker_thread and worker_thread.is_alive():
        worker_thread.join(timeout=2)

    worker_thread = None
    return {"ok": True}


@app.get("/status")
def status():
    return {
        "ok": True,
        "running": worker_thread is not None and worker_thread.is_alive(),
        "video_id": current_video_id,
        "thresholds": {
            "elements": THRESH_ELEMENTS,
            "likely": THRESH_LIKELY
        },
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    global event_loop
    if event_loop is None:
        event_loop = asyncio.get_running_loop()
        print("event_loop set from ws ✅", flush=True)

    q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
    clients.add(q)

    try:
        while True:
            msg = await q.get()
            await websocket.send_text(msg)
    finally:
        clients.discard(q)
        try:
            await websocket.close()
        except Exception:
            pass


@app.on_event("startup")
async def on_startup():
    global event_loop
    event_loop = asyncio.get_running_loop()
