const statusChip = document.getElementById("statusChip");
const player = document.getElementById("player");
const chatFrame = document.getElementById("chatFrame");

const feedLikely = document.getElementById("feedLikely");
const feedElements = document.getElementById("feedElements");
const feedNormal = document.getElementById("feedNormal");

function extractVideoId(input) {
  const s = (input || "").trim();
  if (/^[a-zA-Z0-9_-]{11}$/.test(s)) return s;
  const patterns = [
    /v=([a-zA-Z0-9_-]{11})/,
    /youtu\.be\/([a-zA-Z0-9_-]{11})/,
    /live\/([a-zA-Z0-9_-]{11})/
  ];
  for (const p of patterns) {
    const m = s.match(p);
    if (m && m[1]) return m[1];
  }
  return null;
}

function AssignVideoAndChat(videoId) {
  player.src = `https://www.youtube.com/embed/${videoId}?mute=1&autoplay=0`;
  const domain = location.hostname || "localhost";
  chatFrame.src = `https://www.youtube.com/live_chat?v=${videoId}&embed_domain=${encodeURIComponent(domain)}`;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function CreateMsgCard(evt) {
  const div = document.createElement("div");
  div.className = "msg" + (evt.tier === "SYSTEM" ? " system" : "");

  const score = (typeof evt.p_toxic === "number") ? evt.p_toxic.toFixed(2) : "0.00";

  let tagClass = "good";
  let tagText = "Normal Chat";
  if (evt.tier === "LIKELY_TOXIC") { tagClass = "bad"; tagText = "Likely Toxic Chat"; }
  else if (evt.tier === "TOXIC_ELEMENTS") { tagClass = "mid"; tagText = "Toxic Elements in Chat"; }
  else if (evt.tier === "SYSTEM") { tagClass = "mid"; tagText = "System"; }

  const userLink = evt.links?.user || evt.links?.search_user || "";

  div.innerHTML = `
    <div class="meta">
      <span class="author">${escapeHtml(evt.author || "")}</span>
      <span class="score">TOXIC: ${score}</span>
    </div>
    <div>${escapeHtml(evt.text || "")}</div>
    <div class="tag ${tagClass}">${tagText}</div>
    <div class="actions">
      ${userLink ? `<a class="btnlink" href="${userLink}" target="_blank" rel="noopener">View User</a>` : ""}
    </div>
  `;
  return div;
}

function MsgCardClassification(evt) {
  const card = CreateMsgCard(evt);
  if (evt.tier === "LIKELY_TOXIC") feedLikely.prepend(card);
  else if (evt.tier === "TOXIC_ELEMENTS") feedElements.prepend(card);
  else if (evt.tier === "NORMAL") feedNormal.prepend(card);
  else feedElements.prepend(card);
}

let ws = null;

function connectWS() {
  if (ws && (ws.readyState === 0 || ws.readyState === 1)) return;
  const proto = (location.protocol === "https:") ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => setStatus("Connected");
  ws.onclose = () => setStatus("Disconnected");
  ws.onerror = () => setStatus("WS Error");
  ws.onmessage = (ev) => {
    try {
      const evt = JSON.parse(ev.data);
      MsgCardClassification(evt);
    } catch (e) {}
  };
}

connectWS();

async function startMonitoring() {
  const input = document.getElementById("ytInput").value;
  const vid = extractVideoId(input);
  if (!vid) { alert("Could not extract video ID from input."); return; }

  const res = await fetch("/ConnectBtn", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({stream: input})
  });
  const data = await res.json();
  if (!data.ok) { alert(data.error || "Connect failed"); return; }

  AssignVideoAndChat(data.video_id);
  setStatus("Running");
}

document.getElementById("ConnectBtn").addEventListener("click", startMonitoring);