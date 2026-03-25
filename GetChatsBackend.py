import asyncio
import json
import os
import re
import threading
import time
from typing import Optional, Set

import requests
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
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

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
    device=device,
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
worker_thread: Optional[threading.Thread] = None
stop_flag = threading.Event()
current_video_id: Optional[str] = None


def broadcast_json(payload: str):
    if event_loop is None or len(clients) == 0:
        return

    async def _push():
        for q in list(clients):
            try:
                q.put_nowait(payload)
            except Exception:
                pass

    asyncio.run_coroutine_threadsafe(_push(), event_loop)


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


def get_active_live_chat_id(video_id: str) -> Optional[str]:
    if not YOUTUBE_API_KEY:
        raise RuntimeError("Missing YOUTUBE_API_KEY")

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "liveStreamingDetails",
        "id": video_id,
        "key": YOUTUBE_API_KEY,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    items = data.get("items", [])
    if not items:
        return None

    details = items[0].get("liveStreamingDetails", {})
    return details.get("activeLiveChatId")


def fetch_live_chat_messages(live_chat_id: str, page_token: Optional[str] = None) -> dict:
    if not YOUTUBE_API_KEY:
        raise RuntimeError("Missing YOUTUBE_API_KEY")

    url = "https://www.googleapis.com/youtube/v3/liveChat/messages"
    params = {
        "liveChatId": live_chat_id,
        "part": "id,snippet,authorDetails",
        "maxResults": 200,
        "key": YOUTUBE_API_KEY,
    }

    if page_token:
        params["pageToken"] = page_token

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def chat_worker(video_id: str):
    global current_video_id
    current_video_id = video_id

    try:
        live_chat_id = get_active_live_chat_id(video_id)
        if not live_chat_id:
            current_video_id = None
            return

        next_page_token = None
        if len(seen_ids) > 5000:
        seen_ids.clear()

        while not stop_flag.is_set():
              if len(clients) == 0:
                time.sleep(5)
                continue
                
            data = fetch_live_chat_messages(live_chat_id, next_page_token)

            for item in data.get("items", []):
                msg_id = item.get("id")
                if not msg_id or msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)

                author = item.get("authorDetails", {}).get("displayName", "")
                text = item.get("snippet", {}).get("displayMessage", "")
                ts = item.get("snippet", {}).get("publishedAt", "")

                scores = get_scores(text)
                p_toxic = scores.get("TOXIC", scores.get("LABEL_1", 0.0))
                tier = tier_for_score(p_toxic)

                user_link = f"https://www.youtube.com/results?search_query={author}"

                payload_obj = {
                    "ts": ts,
                    "video_id": video_id,
                    "author": author,
                    "text": text,
                    "p_toxic": float(p_toxic),
                    "tier": tier,
                    "links": {
                        "user": user_link
                    },
                }

                broadcast_json(json.dumps(payload_obj, ensure_ascii=False))

            next_page_token = data.get("nextPageToken")
            wait_ms = data.get("pollingIntervalMillis", 2000)
            time.sleep(max(wait_ms / 1000.0, 2))

    except Exception as e:
        error_payload = {
            "ts": "",
            "video_id": video_id,
            "author": "",
            "text": f"[Backend error] {str(e)}",
            "p_toxic": 0.0,
            "tier": "SYSTEM",
            "links": {"user": ""},
        }
        broadcast_json(json.dumps(error_payload, ensure_ascii=False))
        current_video_id = None


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
        daemon=True,
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
            "likely": THRESH_LIKELY,
        },
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    global event_loop
    if event_loop is None:
        event_loop = asyncio.get_running_loop()

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