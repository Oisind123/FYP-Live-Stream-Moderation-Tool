# Real-Time YouTube Live Chat Moderation Dashboard

A real-time moderation dashboard for YouTube livestream chats.  
This project fetches live chat messages using the YouTube Data API, classifies them with a Toxic-BERT model, and displays them in real time on a dashboard.

## Features

- Monitor YouTube livestream chat in real time
- Classify messages using a pre-trained toxicity-BERT model
- Categorise messages into:
  - Normal
  - Potentially Toxic
  - Toxic
- View the livestream, official YouTube chat, and moderation feed side by side
- Receive live updates through WebSockets
- Simple FastAPI backend and browser-based frontend

## How It Works

1. The user enters a YouTube livestream URL or video ID
2. The frontend sends this input to the FastAPI backend
3. The backend extracts the video ID
4. The backend uses the YouTube Data API to get the `activeLiveChatId`
5. Live chat messages are fetched continuously from YouTube using polling
6. Each message is passed into the toxicity-BERT model
7. A toxicity score is returned and mapped to one of three categories
8. Classified messages are sent to the frontend through a WebSocket
9. The frontend displays the messages instantly in separate columns

## Tech Stack

- Backend: FastAPI, Python
- Frontend: HTML, CSS, JavaScript
- Machine Learning: Hugging Face Transformer, toxicity-BERT
- API: YouTube Data API
- Real-time Communication: WebSockets
- Inference Support: PyTorch