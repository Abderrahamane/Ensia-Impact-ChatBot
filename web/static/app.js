const statusEl = document.getElementById("status");
const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const themeBtn = document.getElementById("themeBtn");
const exportBtn = document.getElementById("exportBtn");
const apiBaseInput = document.getElementById("apiBaseInput");
const saveApiBtn = document.getElementById("saveApiBtn");
const quickPromptsEl = document.getElementById("quickPrompts");
const toastEl = document.getElementById("toast");
const msgCountEl = document.getElementById("msgCount");
const apiModeEl = document.getElementById("apiMode");
const CHAT_STORAGE_KEY = "ensia_web_chat_v1";
const THEME_STORAGE_KEY = "ensia_web_theme";
const API_BASE_KEY = "ensia_api_base";

function getApiBase() {
  const custom = (localStorage.getItem(API_BASE_KEY) || "").trim();
  return custom || "";
}

function apiUrl(path) {
  const base = getApiBase();
  if (!base) return path;
  return base.replace(/\/$/, "") + path;
}

function setOnlineState(isOnline, message) {
  statusEl.textContent = message;
  statusEl.classList.remove("ok", "down");
  statusEl.classList.add(isOnline ? "ok" : "down");
  inputEl.disabled = !isOnline;
  sendBtn.disabled = !isOnline;
}

function setModeLabel(mode) {
  if (!apiModeEl) return;
  apiModeEl.textContent = `Mode: ${mode || "-"}`;
}

function showToast(message) {
  if (!toastEl) return;
  toastEl.textContent = message;
  toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 1800);
}

function saveChatHistory() {
  localStorage.setItem(CHAT_STORAGE_KEY, chatEl.innerHTML);
  updateMessageCount();
}

function loadChatHistory() {
  const html = localStorage.getItem(CHAT_STORAGE_KEY);
  if (html) chatEl.innerHTML = html;
  chatEl.scrollTop = chatEl.scrollHeight;
  updateMessageCount();
}

function updateMessageCount() {
  if (!msgCountEl) return;
  const count = chatEl.querySelectorAll(".msg").length;
  msgCountEl.textContent = `${count} message${count === 1 ? "" : "s"}`;
}

function setTheme(theme) {
  document.body.classList.toggle("light", theme === "light");
  localStorage.setItem(THEME_STORAGE_KEY, theme);
}

function loadTheme() {
  const t = localStorage.getItem(THEME_STORAGE_KEY) || "dark";
  setTheme(t);
}

function addMessage(role, text, sources = []) {
  const wrap = document.createElement("div");
  wrap.className = "msg-wrap";

  const div = document.createElement("div");
  div.className = `msg ${role}`;

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  meta.textContent = role === "user" ? `You · ${ts}` : `Assistant · ${ts}`;
  div.appendChild(meta);

  if (role === "bot") {
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.type = "button";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        showToast("Copied response");
      } catch {
        showToast("Copy failed");
      }
    });
    meta.appendChild(copyBtn);
  }

  const content = document.createElement("div");
  content.textContent = text;
  div.appendChild(content);
  wrap.appendChild(div);

  if (role === "bot" && Array.isArray(sources) && sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    const bits = sources.slice(0, 3).map((s) => {
      const trust = s.trust ? ` (${s.trust})` : "";
      const label = `${s.date || ""} ${s.from || ""}`.trim() || "source";
      const link = (s.links || "").split("|")[0]?.trim();
      if (link && /^https?:\/\//.test(link)) {
        return `<a href="${link}" target="_blank" rel="noopener noreferrer">${label}${trust}</a>`;
      }
      return `${label}${trust}`;
    });
    src.innerHTML = "Sources: " + (bits.length ? bits.join(" | ") : "n/a");
    wrap.appendChild(src);
  }

  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  saveChatHistory();
}

function showTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg-wrap";
  wrap.id = "typingWrap";
  const div = document.createElement("div");
  div.className = "msg bot";
  div.innerHTML = "Thinking <span class=\"typing\"><span></span><span></span><span></span></span>";
  wrap.appendChild(div);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function hideTyping() {
  const t = document.getElementById("typingWrap");
  if (t) t.remove();
}

function clearChat() {
  chatEl.innerHTML = "";
  localStorage.removeItem(CHAT_STORAGE_KEY);
  updateMessageCount();
}

async function loadApiInfo() {
  try {
    const res = await fetch(apiUrl("/api/info"), { method: "GET" });
    if (!res.ok) return;
    const data = await res.json();
    setModeLabel(data.backend || "-");
  } catch {
    setModeLabel("-");
  }
}

async function checkHealth() {
  try {
    const res = await fetch(apiUrl("/api/health"), { method: "GET" });
    const data = await res.json();
    if (data.ok) {
      setOnlineState(true, `Server is online (backend: ${data.backend}).`);
      setModeLabel(data.backend || "-");
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
  showTyping();
  const startedAt = Date.now();

  try {
    const res = await fetch(apiUrl("/api/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    hideTyping();
    addMessage("bot", data.answer || "No response.", data.sources || []);
    if (data.mode) {
      setModeLabel(data.mode);
    }
    const elapsedMs = Date.now() - startedAt;
    showToast(`Answered in ${elapsedMs} ms`);
  } catch {
    hideTyping();
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
  newChatBtn.addEventListener("click", () => {
    clearChat();
    showToast("Chat reset");
  });
}

if (themeBtn) {
  themeBtn.addEventListener("click", () => {
    const next = document.body.classList.contains("light") ? "dark" : "light";
    setTheme(next);
  });
}

if (exportBtn) {
  exportBtn.addEventListener("click", () => {
    const payload = {
      exported_at: new Date().toISOString(),
      endpoint: getApiBase() || window.location.origin,
      transcript_text: chatEl.innerText,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ensia_chat_export.json";
    a.click();
    URL.revokeObjectURL(url);
  });
}

if (apiBaseInput) {
  apiBaseInput.value = localStorage.getItem(API_BASE_KEY) || "";
}

if (saveApiBtn) {
  saveApiBtn.addEventListener("click", () => {
    localStorage.setItem(API_BASE_KEY, (apiBaseInput.value || "").trim());
    showToast("API endpoint saved");
    checkHealth();
  });
}

if (quickPromptsEl) {
  quickPromptsEl.addEventListener("click", (e) => {
    const target = e.target;
    if (!(target instanceof HTMLElement)) return;
    if (!target.classList.contains("prompt-chip")) return;
    const q = target.getAttribute("data-q") || "";
    inputEl.value = q;
    inputEl.dispatchEvent(new Event("input"));
    inputEl.focus();
  });
}

loadTheme();
loadChatHistory();
loadApiInfo();
checkHealth();
setInterval(checkHealth, 10000);

