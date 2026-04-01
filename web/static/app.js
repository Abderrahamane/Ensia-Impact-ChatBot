const statusEl = document.getElementById("status");
const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const CHAT_STORAGE_KEY = "ensia_web_chat_v1";

function setOnlineState(isOnline, message) {
  statusEl.textContent = message;
  statusEl.classList.remove("ok", "down");
  statusEl.classList.add(isOnline ? "ok" : "down");
  inputEl.disabled = !isOnline;
  sendBtn.disabled = !isOnline;
}

function saveChatHistory() {
  localStorage.setItem(CHAT_STORAGE_KEY, chatEl.innerHTML);
}

function loadChatHistory() {
  const html = localStorage.getItem(CHAT_STORAGE_KEY);
  if (html) chatEl.innerHTML = html;
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(role, text, sources = []) {
  const wrap = document.createElement("div");
  wrap.className = "msg-wrap";

  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  wrap.appendChild(div);

  if (role === "bot" && Array.isArray(sources) && sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    src.textContent = "Sources: " + sources.slice(0, 2).map((s) => `${s.date || ""} ${s.from || ""}`).join(" | ");
    wrap.appendChild(src);
  }

  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  saveChatHistory();
}

function clearChat() {
  chatEl.innerHTML = "";
  localStorage.removeItem(CHAT_STORAGE_KEY);
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health", { method: "GET" });
    const data = await res.json();
    if (data.ok) {
      setOnlineState(true, `Server is online (backend: ${data.backend}).`);
    } else {
      setOnlineState(false, "Server is off, try later.");
    }
  } catch {
    setOnlineState(false, "Server is off, try later.");
  }
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  addMessage("user", message);
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    addMessage("bot", data.answer || "No response.", data.sources || []);
  } catch {
    addMessage("bot", "Server is down, please try later.");
  } finally {
    sendBtn.disabled = inputEl.disabled;
  }
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

if (newChatBtn) {
  newChatBtn.addEventListener("click", clearChat);
}

loadChatHistory();
checkHealth();
setInterval(checkHealth, 10000);

