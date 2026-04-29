"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

type JwtPayload = {
  sub?: string; username?: string; organization?: string; role?: string; exp?: number;
};

type ChatSource = {
  source: string; page?: number | null; row?: number | null;
  tenant?: string | null; metadata?: Record<string, unknown>;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  sources?: ChatSource[];
  confidence?: number;
  confidence_label?: string;
  conflict_detected?: boolean;
  latency_ms?: number;
  ttft_ms?: number;
  query_id?: string;
  abstained?: boolean;
  abstain_reason?: string;
  feedback?: "up" | "down" | null;
  error?: boolean;
  retryPrompt?: string;
};

type ChatInterfaceProps = { apiBaseUrl?: string };

const CONFIDENCE_VERIFIED_THRESHOLD = 0.4;

const SUGGESTIONS = [
  { icon: "🕐", text: "What are the core hours?" },
  { icon: "🌴", text: "How do I request PTO?" },
  { icon: "🏥", text: "Tell me about health benefits" },
];

function decodeJwt(token: string): JwtPayload | null {
  try {
    const p = token.split(".")[1];
    if (!p) return null;
    const b = p.replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(b.padEnd(b.length + ((4 - (b.length % 4)) % 4), "="))) as JwtPayload;
  } catch { return null; }
}

const STORAGE_KEY = "dayone_messages";

export default function ChatInterface({ apiBaseUrl }: ChatInterfaceProps) {
  const router = useRouter();
  const apiRoot = useMemo(
    () => apiBaseUrl ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000",
    [apiBaseUrl],
  );

  const [token, setToken] = useState<string | null>(null);
  const [profile, setProfile] = useState<JwtPayload | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState("");
  const [sending, setSending] = useState(false);
  const [minimalMode, setMinimalMode] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const rehydratedRef = useRef(false);

  useEffect(() => {
    const isMinimal = localStorage.getItem("dayone_minimal_mode") === "true";
    setMinimalMode(isMinimal);
    if (isMinimal) document.body.classList.add("minimal-mode");
  }, []);

  const toggleMinimalMode = () => {
    const newVal = !minimalMode;
    setMinimalMode(newVal);
    localStorage.setItem("dayone_minimal_mode", String(newVal));
    if (newVal) {
      document.body.classList.add("minimal-mode");
    } else {
      document.body.classList.remove("minimal-mode");
    }
  };

  // Auth guard + rehydrate conversation from localStorage
  useEffect(() => {
    if (rehydratedRef.current) return;
    const stored = localStorage.getItem("dayone_token");
    if (!stored) { router.replace("/login"); return; }
    const decoded = decodeJwt(stored);
    if (!decoded || decoded.role === "admin") {
      router.replace(decoded?.role === "admin" ? "/admin" : "/login"); return;
    }
    setToken(stored);
    setProfile(decoded);
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) setMessages(JSON.parse(saved) as ChatMessage[]);
    } catch { /* ignore */ }
    rehydratedRef.current = true;
  }, [router]);

  // Persist messages to localStorage on change
  useEffect(() => {
    if (messages.length > 0)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  const appendMessages = (...nextMessages: ChatMessage[]) => {
    setMessages(prev => [...prev, ...nextMessages]);
  };

  const replaceLastMessage = (updater: (current: ChatMessage) => ChatMessage) => {
    setMessages(prev => {
      if (prev.length === 0) return prev;
      const next = [...prev];
      next[next.length - 1] = updater(next[next.length - 1]);
      return next;
    });
  };

  async function sendPrompt(text: string) {
    const trimmed = text.trim();
    if (!trimmed || !token || sending) return;

    const userMsg: ChatMessage = { role: "user", content: trimmed };
    // Append user message + streaming placeholder
    appendMessages(userMsg, { role: "assistant", content: "", streaming: true });
    setPrompt("");
    setSending(true);

    try {
      const response = await fetch(`${apiRoot}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body: JSON.stringify({ prompt: trimmed }),
      });

      console.log(response);

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let meta: Partial<ChatMessage> = {};
      let accumulated = "";
      let streamError: string | null = null;
      let sawDone = false;

      const processEventBlock = async (block: string) => {
        const dataLines = block
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim());
        if (dataLines.length === 0) return;

        try {
          const evt = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
          if (evt.type === "meta") {
            meta = {
              confidence: evt.confidence as number,
              confidence_label: evt.confidence_label as string,
              conflict_detected: evt.conflict_detected as boolean,
              abstained: evt.abstained as boolean,
              abstain_reason: evt.abstain_reason as string,
              sources: evt.sources as ChatSource[],
            };
            return;
          }

          if (evt.type === "ttft") {
            meta.ttft_ms = evt.ttft_ms as number;
            return;
          }

          if (evt.type === "token") {
            accumulated += String(evt.content ?? "").replace(/\\n/g, "\n").replace(/\\"/g, '"');
            
            // Artificial cadence: slight delay between chunks for "rhythm"
            // Adaptive: short responses feel human, long ones feel fast.
            const baseDelay = accumulated.length > 300 ? 5 : 15;
            const randomFactor = accumulated.length > 300 ? 10 : 20;
            await new Promise(r => setTimeout(r, baseDelay + Math.random() * randomFactor));

            replaceLastMessage(current => ({
              ...current,
              content: accumulated,
              ...meta,
            }));
            return;
          }

          if (evt.type === "done") {
            meta.latency_ms = evt.latency_ms as number;
            meta.query_id = evt.query_id as string;
            sawDone = true;
            return;
          }

          if (evt.type === "error") {
            streamError = String(evt.detail ?? "Stream failed.");
          }
        } catch {
          // Ignore malformed event blocks so one bad chunk does not break the whole stream.
        }
      };

      const nextEventBoundary = (value: string): { index: number; length: number } | null => {
        const crlfIdx = value.indexOf("\r\n\r\n");
        const lfIdx = value.indexOf("\n\n");
        if (crlfIdx < 0 && lfIdx < 0) return null;
        if (crlfIdx >= 0 && (lfIdx < 0 || crlfIdx <= lfIdx)) return { index: crlfIdx, length: 4 };
        return { index: lfIdx, length: 2 };
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let boundary = nextEventBoundary(buf);
        while (boundary) {
          const block = buf.slice(0, boundary.index);
          buf = buf.slice(boundary.index + boundary.length);
          await processEventBlock(block);
          if (streamError || sawDone) break;
          boundary = nextEventBoundary(buf);
        }

        if (streamError || sawDone) {
          try {
            await reader.cancel();
          } catch {
            // Ignore cancellation errors after terminal events.
          }
          break;
        }
      }

      if (!streamError && buf.trim()) {
        await processEventBlock(buf);
      }

      if (streamError) {
        throw new Error(streamError);
      }

      replaceLastMessage(current => ({
        ...current,
        streaming: false,
        ...meta,
        content: accumulated || "I do not have that information in the current HR files. Please contact HR.",
      }));

    } catch (err) {
      replaceLastMessage(() => ({
        role: "assistant",
        content: err instanceof Error ? err.message : "Request failed.",
        streaming: false,
        error: true,
        retryPrompt: trimmed,
      }));
    } finally {
      setSending(false);
    }
  }

  async function submitFeedback(msgIdx: number, rating: "up" | "down") {
    const msg = messages[msgIdx];
    if (!msg || !token || msg.feedback) return;
    setMessages(prev => {
      const next = [...prev];
      next[msgIdx] = { ...next[msgIdx], feedback: rating };
      return next;
    });
    try {
      await fetch(`${apiRoot}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body: JSON.stringify({
          query_id: msg.query_id ?? "unknown",
          rating,
          sources: (msg.sources ?? []).map(s => s.source),
          confidence: msg.confidence ?? 0,
          query: messages[msgIdx - 1]?.content ?? "",
        }),
      });
    } catch { /* non-critical */ }
  }

  function signOut() {
    localStorage.removeItem("dayone_token");
    localStorage.removeItem("dayone_profile");
    localStorage.removeItem(STORAGE_KEY);
    router.replace("/login");
  }

  if (!token) return (
    <main style={{ minHeight: "100vh", background: "#020617", display: "grid", placeItems: "center" }}>
      <div style={{ fontSize: "0.875rem", color: "#475569" }}>Loading workspace...</div>
    </main>
  );

  const organization = profile?.organization ?? "Unknown";
  const usernameDisplay = profile?.username ?? "User";
  const avatarInitial = usernameDisplay[0]?.toUpperCase() ?? "U";

  const getConfidenceLabel = (confidence?: number) => {
    if (confidence === undefined) return null;
    return confidence >= CONFIDENCE_VERIFIED_THRESHOLD ? "Verified" : "Moderate";
  };

  const getConfidenceTone = (confidence?: number) => {
    if (confidence === undefined) return "moderate";
    return confidence >= CONFIDENCE_VERIFIED_THRESHOLD ? "verified" : "moderate";
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .chat-root { position:relative; overflow:hidden; font-family:'Inter',sans-serif; min-height:100vh; background:
          radial-gradient(circle at top left, rgba(56,189,248,0.16), transparent 34%),
          radial-gradient(circle at 85% 10%, rgba(14,165,233,0.14), transparent 28%),
          radial-gradient(circle at bottom right, rgba(15,23,42,0.9), #020617 60%);
          color:#f1f5f9; }
        .chat-root::before { content:""; position:fixed; inset:0; pointer-events:none; background:radial-gradient(circle at 50% 0%, rgba(255,255,255,0.05), transparent 35%), radial-gradient(circle at 0% 100%, rgba(56,189,248,0.06), transparent 30%); opacity:0.85; }
        @keyframes gradientDrift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
        .monogram {
          width: 56px;
          height: 56px;
          border-radius: 16px;
          background: rgba(56, 189, 248, 0.1);
          border: 1px solid rgba(56, 189, 248, 0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 1.25rem;
          font-size: 1.25rem;
          font-weight: 700;
          color: #38bdf8;
          letter-spacing: -0.04em;
          box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
        }
        .chat-sidebar { position:fixed; left:0; top:0; width:280px; height:100vh; display:flex; flex-direction:column; border-right:1px solid rgba(255,255,255,0.08); background:rgba(10,18,35,0.7); backdrop-filter:blur(24px) saturate(140%); padding:1.5rem 1.25rem; z-index:20; box-shadow:0 24px 80px rgba(0,0,0,0.35); }
        .user-card { display:flex; align-items:center; gap:0.75rem; padding:0.875rem; border-radius:16px; border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.04); margin-bottom:1rem; backdrop-filter:blur(12px); }
        .user-avatar { width:38px; height:38px; border-radius:50%; background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.35); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.95rem; color:#38bdf8; flex-shrink:0; }
        .org-badge { border-radius:14px; border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.04); padding:1rem; margin-bottom:auto; backdrop-filter:blur(12px); }
        .nav-item { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; border-radius: 12px; color: #94a3b8; text-decoration: none; font-size: 0.875rem; font-weight: 500; transition: all 0.2s; margin-bottom: 0.25rem; border: 1px solid transparent; cursor: pointer; }
        .nav-item:hover { background: rgba(56, 189, 248, 0.05); color: #f1f5f9; }
        .nav-item-active { background: rgba(56, 189, 248, 0.1); color: #38bdf8; border-color: rgba(56, 189, 248, 0.2); }
        .signout-btn { width:100%; border-radius:14px; border:1px solid rgba(56,189,248,0.25); background:linear-gradient(135deg, rgba(14,165,233,0.18), rgba(59,130,246,0.12)); padding:0.75rem 1rem; font-size:0.875rem; font-weight:600; font-family:'Inter',sans-serif; color:#e2e8f0; cursor:pointer; transition:transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease; margin-top:1rem; box-shadow:0 10px 28px rgba(0,0,0,0.22); }
        .signout-btn:hover { border-color:rgba(56,189,248,0.45); transform:translateY(-1px) scale(1.01); box-shadow:0 16px 36px rgba(14,165,233,0.16); color:#ffffff; }
        .chat-main { margin-left:280px; min-height:100vh; display:flex; flex-direction:column; }
        .chat-main.sidebar-closed { margin-left:0; }
        .chat-header { border-bottom:1px solid rgba(255,255,255,0.08); background:rgba(2,6,23,0.72); backdrop-filter:blur(24px) saturate(140%); padding:1.5rem 2rem; position:sticky; top:0; z-index:10; }
        .chat-messages { flex:1; padding:2rem; padding-bottom:7rem; overflow-y:auto; }
        .welcome-title { font-size:2.25rem; font-weight:700; color:#f8fafc; letter-spacing:-0.04em; margin:0 0 0.75rem; }
        .suggestion-card { border-radius:16px; border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.04); padding:1rem; text-align:left; font-size:0.875rem; font-family:'Inter',sans-serif; color:#cbd5e1; cursor:pointer; transition:transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease, background 0.2s ease; display:flex; flex-direction:column; gap:0.4rem; backdrop-filter:blur(12px); }
        .suggestion-card:hover { border-color:rgba(56,189,248,0.35); background:rgba(255,255,255,0.06); transform:translateY(-2px) scale(1.01); box-shadow:0 14px 34px rgba(0,0,0,0.22); }
        @keyframes messageIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
        .message-bubble { animation:messageIn 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
        .user-bubble { max-width:80%; border-radius:22px 22px 8px 22px; background:linear-gradient(135deg, rgba(14,165,233,0.95), rgba(59,130,246,0.92)); padding:0.925rem 1.15rem; color:white; font-weight:500; box-shadow:0 18px 40px rgba(14,165,233,0.22); border:1px solid rgba(255,255,255,0.12); }
        .assistant-bubble { max-width:80%; border-radius:22px 22px 22px 8px; border:1px solid rgba(255,255,255,0.1); background:rgba(255,255,255,0.05); padding:0.925rem 1.15rem; color:#f1f5f9; line-height:1.7; box-shadow:0 18px 36px rgba(0,0,0,0.28); backdrop-filter:blur(18px) saturate(150%); }
        .assistant-bubble.error-bubble { border-color:rgba(244,63,94,0.35); background:rgba(244,63,94,0.09); }
        @keyframes cursor-blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .streaming-cursor { display:inline-block; width:2px; height:1em; background:#38bdf8; margin-left:2px; vertical-align:text-bottom; animation:cursor-blink 0.8s ease infinite; }
        .meta-bar { margin-top:0.8rem; display:flex; align-items:center; flex-wrap:wrap; gap:0.6rem; }
        .conf-badge { display:inline-flex; align-items:center; gap:0.35rem; font-size:0.72rem; font-weight:700; padding:0.4rem 0.8rem; border-radius:999px; letter-spacing:0.02em; transition:all 0.2s ease; box-shadow:0 10px 24px rgba(0,0,0,0.18); }
        .conf-verified { background:linear-gradient(135deg, rgba(34,197,94,0.18), rgba(16,185,129,0.12)); color:#bbf7d0; border:1px solid rgba(34,197,94,0.28); }
        .conf-moderate { background:linear-gradient(135deg, rgba(234,179,8,0.16), rgba(245,158,11,0.1)); color:#fde68a; border:1px solid rgba(234,179,8,0.28); }
        .abstain-bubble { border: 1px solid rgba(244, 63, 94, 0.4) !important; background: rgba(244, 63, 94, 0.08) !important; box-shadow: 0 4px 12px rgba(244, 63, 94, 0.1) !important; }
        .abstain-icon { font-size: 1.6rem; margin-bottom: 0.5rem; display: block; filter: drop-shadow(0 0 4px rgba(244, 63, 94, 0.2)); }
        .abstain-title { font-weight: 700; color: #fda4af; margin-bottom: 0.3rem; display: block; letter-spacing: -0.01em; }
        .verification-details { margin-top:0.8rem; border-radius:12px; border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.04); padding:0.6rem 0.85rem; backdrop-filter:blur(12px); }
        .verification-summary { cursor:pointer; font-size:0.78rem; font-weight:600; color:#94a3b8; user-select:none; list-style:none; }
        .verification-summary::-webkit-details-marker { display:none; }
        .verification-body { margin-top:0.65rem; display:grid; gap:0.5rem; }
        .verification-item { display:flex; align-items:flex-start; justify-content:space-between; gap:0.75rem; padding:0.5rem 0.65rem; border-radius:10px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); }
        .verification-label { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; font-weight:600; }
        .verification-value { font-size:0.78rem; color:#e2e8f0; text-align:right; }
        .feedback-row { display:flex; align-items:center; gap:0.35rem; margin-top:0.5rem; }
        .thumb-btn { background:none; border:1px solid rgba(51,65,85,0.5); border-radius:8px; padding:0.25rem 0.55rem; font-size:0.85rem; cursor:pointer; transition:all 0.15s; color:#64748b; }
        .thumb-btn:hover:not(:disabled) { border-color:rgba(56,189,248,0.4); color:#38bdf8; background:rgba(56,189,248,0.05); }
        .thumb-btn.voted-up { border-color:rgba(34,197,94,0.5); color:#86efac; background:rgba(34,197,94,0.08); }
        .thumb-btn.voted-down { border-color:rgba(244,63,94,0.5); color:#fda4af; background:rgba(244,63,94,0.08); }
        .thumb-btn:disabled { opacity:0.6; cursor:default; }
        .retry-btn { font-size:0.75rem; padding:0.32rem 0.8rem; border-radius:999px; border:1px solid rgba(56,189,248,0.35); background:linear-gradient(135deg, rgba(14,165,233,0.18), rgba(59,130,246,0.12)); color:#e0f2fe; cursor:pointer; font-family:'Inter',sans-serif; transition:transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease; box-shadow:0 10px 22px rgba(0,0,0,0.18); }
        .retry-btn:hover { transform:translateY(-1px) scale(1.01); box-shadow:0 14px 28px rgba(14,165,233,0.15); }
        .sources-details { margin-top:0.875rem; border-radius:12px; border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.04); padding:0.75rem 1rem; backdrop-filter:blur(12px); }
        .sources-summary { font-size:0.8rem; font-weight:600; color:#38bdf8; user-select:none; margin-bottom:0.45rem; }
        .source-item { border-radius:10px; border:1px solid rgba(255,255,255,0.06); background:rgba(255,255,255,0.04); padding:0.5rem 0.75rem; font-size:0.8rem; margin-top:0.4rem; }
        @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }
        .typing-dot { width:7px; height:7px; border-radius:50%; background:#38bdf8; animation:bounce 1.2s ease-in-out infinite; }
        .chat-input-wrap { position:sticky; bottom:0; border-top:1px solid rgba(255,255,255,0.08); background:rgba(2,6,23,0.74); backdrop-filter:blur(24px) saturate(140%); padding:1.25rem 2rem; }
        .chat-input-inner { max-width:800px; margin:0 auto; display:flex; align-items:flex-end; gap:0.75rem; border-radius:20px; border:1px solid rgba(255,255,255,0.09); background:rgba(255,255,255,0.04); padding:0.75rem 0.75rem 0.75rem 1.125rem; box-shadow:0 18px 44px rgba(0,0,0,0.28); transition:border-color 0.2s, transform 0.2s; backdrop-filter:blur(16px) saturate(150%); }
        .chat-input-inner:focus-within { border-color:rgba(56,189,248,0.35); transform:translateY(-1px); }
        .chat-textarea { flex:1; background:transparent; border:none; outline:none; resize:none; font-size:0.9rem; font-family:'Inter',sans-serif; color:#f1f5f9; line-height:1.5; max-height:160px; min-height:24px; overflow-y:auto; padding:0; }
        .chat-textarea::placeholder { color:#475569; }
        .send-btn { border-radius:14px; background:linear-gradient(135deg, #38bdf8, #0ea5e9 55%, #3b82f6); border:none; padding:0.62rem 1.1rem; font-size:0.875rem; font-weight:700; font-family:'Inter',sans-serif; color:#020617; cursor:pointer; transition:transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease; flex-shrink:0; box-shadow:0 16px 30px rgba(14,165,233,0.25); }
        .send-btn:hover:not(:disabled) { transform:translateY(-1px) scale(1.03); box-shadow:0 20px 36px rgba(56,189,248,0.35); filter:brightness(1.04); }
        .send-btn:disabled { opacity:0.55; cursor:not-allowed; }
        .clear-link { font-size:0.72rem; color:#475569; background:none; border:none; cursor:pointer; font-family:'Inter',sans-serif; padding:0; transition:color 0.15s; }
        .clear-link:hover { color:#94a3b8; }
        .sidebar-toggle-btn { position:fixed; top:1rem; left:1rem; z-index:60; display:inline-flex; align-items:center; justify-content:center; width:42px; height:42px; border-radius:12px; border:1px solid rgba(56,189,248,0.28); background:rgba(2,6,23,0.75); backdrop-filter:blur(16px) saturate(140%); color:#e2e8f0; cursor:pointer; box-shadow:0 16px 34px rgba(0,0,0,0.34); transition:transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease; }
        .sidebar-toggle-btn:hover { border-color:rgba(56,189,248,0.5); transform:translateY(-1px) scale(1.03); box-shadow:0 20px 38px rgba(0,0,0,0.38); }
        .sidebar-close-btn { align-self:flex-end; margin-bottom:1rem; width:34px; height:34px; border-radius:10px; border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.05); color:#cbd5e1; cursor:pointer; backdrop-filter:blur(12px); transition:transform 0.2s ease, border-color 0.2s ease, background 0.2s ease; }
        .sidebar-close-btn:hover { border-color:rgba(56,189,248,0.35); background:rgba(255,255,255,0.08); transform:scale(1.04); }
      `}</style>

      <main className="chat-root">
        {!sidebarOpen && (
          <button
            type="button"
            className="sidebar-toggle-btn"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open sidebar"
          >
            ☰
          </button>
        )}

        {/* Sidebar */}
        {sidebarOpen && (
          <aside className="chat-sidebar">
            <button
              type="button"
              className="sidebar-close-btn"
              onClick={() => setSidebarOpen(false)}
              aria-label="Close sidebar"
            >
              ×
            </button>

            {/* Brand Monogram */}
            <div className="flex items-center gap-3 mb-8 px-2">
              <div className="monogram" style={{ margin: 0, width: "42px", height: "42px", borderRadius: "12px", fontSize: "1rem" }}>D1</div>
              <span style={{ fontWeight: 700, fontSize: "1.1rem", color: "#f8fafc", letterSpacing: "-0.02em" }}>DayOne AI</span>
            </div>

            <div className="user-card">
              <div className="user-avatar" aria-hidden="true">{avatarInitial}</div>
              <div style={{ minWidth: 0 }}>
                <p style={{ margin: 0, fontSize: "0.875rem", fontWeight: 600, color: "#f1f5f9", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{usernameDisplay}</p>
                <p style={{ margin: 0, fontSize: "0.75rem", color: "#64748b" }}>Employee</p>
              </div>
            </div>

            <nav style={{ flex: 1, marginTop: "1rem" }}>
              <div className="nav-item nav-item-active">
                <span>💬</span> Chat
              </div>
              {profile?.role === "admin" && (
                <div className="nav-item" onClick={() => router.push("/admin")} style={{ cursor: "pointer" }}>
                  <span>📊</span> Dashboard
                </div>
              )}
            </nav>

            <div className="org-badge">
              <p style={{ margin: 0, fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.2em", color: "#475569", fontWeight: 600 }}>Organization</p>
              <p style={{ margin: "0.4rem 0 0", fontSize: "1rem", fontWeight: 600, color: "#f1f5f9" }}>{organization}</p>
            </div>

            {/* Minimal Mode Toggle */}
            <div style={{ padding: "1rem 0.5rem 0.5rem" }}>
              <label style={{ display: "flex", alignItems: "center", gap: "0.75rem", cursor: "pointer", color: "#64748b", fontSize: "0.8rem" }}>
                <input 
                  type="checkbox" 
                  checked={minimalMode} 
                  onChange={toggleMinimalMode} 
                  style={{ width: "16px", height: "16px" }}
                />
                Minimal Mode
              </label>
            </div>

            <button onClick={signOut} className="signout-btn" aria-label="Sign out">Sign out</button>
          </aside>
        )}

        {/* Main */}
        <section className={`chat-main${sidebarOpen ? "" : " sidebar-closed"}`}>
          <header className="chat-header">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <p style={{ margin: 0, fontSize: "0.75rem", color: "#38bdf8", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.1em" }}>DayOne AI</p>
                <h1 style={{ margin: "0.25rem 0 0", fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", letterSpacing: "-0.03em" }}>HR Policy Assistant</h1>
              </div>
              <div style={{ padding: "0.5rem 1rem", borderRadius: "12px", background: "rgba(56, 189, 248, 0.05)", border: "1px solid rgba(56, 189, 248, 0.1)", textAlign: "right" }}>
                <p style={{ margin: 0, fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#64748b" }}>Secure Tenant</p>
                <p style={{ margin: 0, fontSize: "0.875rem", fontWeight: 600, color: "#38bdf8" }}>{organization}</p>
              </div>
            </div>
          </header>

          <div className="chat-messages">
            {messages.length === 0 ? (
              <div style={{ display: "grid", minHeight: "55vh", placeItems: "center" }}>
                <div style={{ maxWidth: "640px", textAlign: "center" }}>
                  <h2 className="welcome-title">Welcome to DayOne AI</h2>
                  <p style={{ fontSize: "0.9rem", color: "#64748b", lineHeight: 1.7, margin: "0 auto 2rem", maxWidth: "480px" }}>
                    Ask about onboarding, benefits, PTO, and company policy.
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "0.75rem" }}>
                    {SUGGESTIONS.map(item => (
                      <button key={item.text} onClick={() => void sendPrompt(item.text)} className="suggestion-card" aria-label={`Ask: ${item.text}`}>
                        <span style={{ fontSize: "1.5rem" }}>{item.icon}</span>
                        <span>{item.text}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ maxWidth: "800px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "1.25rem" }}>
                {messages.map((msg, idx) =>
                  msg.role === "user" ? (
                    <div key={idx} className="message-bubble" style={{ display: "flex", justifyContent: "flex-end" }}>
                      <div className="user-bubble">{msg.content}</div>
                    </div>
                  ) : (
                    <div key={idx} className="message-bubble" style={{ display: "flex", justifyContent: "flex-start" }}>
                      <div className={`assistant-bubble${msg.error ? " error-bubble" : ""}${msg.abstained ? " abstain-bubble" : ""}`}>
                        {/* Abstention view */}
                        {msg.abstained ? (
                          <div className="animate-fade-in">
                            <span className="abstain-icon">🛡️</span>
                            <span className="abstain-title">Information Guard</span>
                            <p style={{ margin: "0", fontSize: "0.9rem", color: "#cbd5e1" }}>
                              I've checked your HR files, but I cannot find a sufficiently high-confidence answer. 
                              I've abstained from answering to prevent potential misinformation.
                            </p>
                          </div>
                        ) : (
                          /* Normal content with streaming cursor */
                          <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                            {msg.content}
                            {msg.streaming && <span className="streaming-cursor" aria-hidden="true" />}
                          </p>
                        )}

                        {/* Meta bar — confidence */}
                        {!msg.streaming && msg.confidence !== undefined && (
                          <div className="meta-bar">
                            <span className={`conf-badge conf-${getConfidenceTone(msg.confidence)}`}>
                              Confidence: {getConfidenceLabel(msg.confidence)}
                            </span>
                          </div>
                        )}

                        {/* Verification details */}
                        {!msg.streaming && (
                          <details className="verification-details">
                            <summary className="verification-summary">More details</summary>
                            <div className="verification-body">
                              {msg.confidence !== undefined && (
                                <div className="verification-item">
                                  <span className="verification-label">Confidence score</span>
                                  <span className="verification-value">{(msg.confidence * 100).toFixed(0)}%</span>
                                </div>
                              )}
                              {msg.conflict_detected && (
                                <div className="verification-item">
                                  <span className="verification-label">Verification note</span>
                                  <span className="verification-value">Additional verification recommended</span>
                                </div>
                              )}
                              {msg.abstained && msg.abstain_reason && (
                                <div className="verification-item">
                                  <span className="verification-label">Justification</span>
                                  <span className="verification-value">{msg.abstain_reason}</span>
                                </div>
                              )}
                              {(msg.ttft_ms !== undefined || msg.latency_ms !== undefined) && (
                                <div className="verification-item">
                                  <span className="verification-label">Debug info</span>
                                  <span className="verification-value">
                                    {msg.ttft_ms !== undefined ? `TTFT ${msg.ttft_ms.toFixed(0)} ms` : null}
                                    {msg.ttft_ms !== undefined && msg.latency_ms !== undefined ? " · " : null}
                                    {msg.latency_ms !== undefined ? `${msg.latency_ms.toFixed(0)} ms total` : null}
                                  </span>
                                </div>
                              )}
                            </div>
                          </details>
                        )}

                        {/* Retry button on error */}
                        {msg.error && msg.retryPrompt && (
                          <div style={{ marginTop: "0.75rem" }}>
                            <button className="retry-btn" onClick={() => void sendPrompt(msg.retryPrompt!)}>↻ Retry</button>
                          </div>
                        )}

                        {/* Sources */}
                        {!msg.streaming && msg.sources && msg.sources.length > 0 && (
                          <div className="sources-details">
                            <div className="sources-summary">📎 Sources ({msg.sources.length})</div>
                            {msg.sources.map((s, si) => (
                              <div key={si} className="source-item">
                                <div style={{ fontWeight: 600, color: "#e2e8f0" }}>{s.source}</div>
                                <div style={{ fontSize: "0.72rem", color: "#64748b", marginTop: "0.15rem" }}>
                                  {s.page != null ? `Page ${s.page}` : null}{s.row != null ? ` Row ${s.row}` : null}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Feedback thumbs */}
                        {!msg.streaming && !msg.error && msg.query_id && (
                          <div className="feedback-row">
                            <span style={{ fontSize: "0.72rem", color: "#475569" }}>Helpful?</span>
                            <button
                              className={`thumb-btn${msg.feedback === "up" ? " voted-up" : ""}`}
                              disabled={!!msg.feedback}
                              onClick={() => void submitFeedback(idx, "up")}
                              aria-label="Mark as helpful"
                            >👍</button>
                            <button
                              className={`thumb-btn${msg.feedback === "down" ? " voted-down" : ""}`}
                              disabled={!!msg.feedback}
                              onClick={() => void submitFeedback(idx, "down")}
                              aria-label="Mark as unhelpful"
                            >👎</button>
                          </div>
                        )}
                      </div>
                    </div>
                  )
                )}

                {/* Typing indicator (before first token) */}
                {sending && messages[messages.length - 1]?.content === "" && (
                  <div className="message-bubble" style={{ display: "flex", justifyContent: "flex-start" }}>
                    <div className="assistant-bubble" style={{ padding: "1rem 1.25rem" }}>
                      <div style={{ display: "flex", gap: "5px", alignItems: "center" }}>
                        {[0, 1, 2].map(i => (
                          <span key={i} className="typing-dot" style={{ animationDelay: `${i * 0.18}s` }} aria-hidden="true" />
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                <div ref={bottomRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <form onSubmit={(e: FormEvent<HTMLFormElement>) => { e.preventDefault(); void sendPrompt(prompt); }} className="chat-input-wrap">
            <div style={{ maxWidth: "800px", margin: "0 auto 0.4rem", display: "flex", justifyContent: "flex-end" }}>
              {messages.length > 0 && (
                <button type="button" className="clear-link" onClick={() => { setMessages([]); localStorage.removeItem(STORAGE_KEY); }}>
                  Clear conversation
                </button>
              )}
            </div>
            <div className="chat-input-inner">
              <textarea
                id="chat-input"
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void sendPrompt(prompt); } }}
                placeholder="Ask an HR policy question… (Enter to send)"
                className="chat-textarea"
                rows={1}
                aria-label="Chat message input"
              />
              <button type="submit" id="chat-send-btn" disabled={sending || !prompt.trim()} aria-label="Send message" className="send-btn">
                {sending ? "…" : "Send"}
              </button>
            </div>
          </form>
        </section>
      </main>
    </>
  );
}
