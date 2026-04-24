// Student Learning Assistant — chat UI with student switcher and KaTeX.
// Streams SSE from /api/chat. Events: session | text | tool_use | tool_result | done | error.

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const form = $("composer");
const input = $("input");
const sendBtn = $("send");
const studentCard = $("student-card");
const studentSelect = $("student-select");
const resetBtn = $("reset-btn");
const emptyEl = $("empty");
const emptyTitleEl = $("empty-title");
const emptySuggestionsEl = $("empty-suggestions");

let history = [];
let streaming = false;
let materialsCache = {};
let messagesInner;
let currentStudentId = null;
let currentProfile = null;

const TOOL_VERBS = {
  get_weak_topics:           { running: "Checking weak areas",             done: "Checked weak areas" },
  get_upcoming_tests:        { running: "Looking up upcoming tests",       done: "Fetched upcoming tests" },
  recommend_study_material:  { running: "Searching study library",         done: "Found study materials" },
  plan_study_week:           { running: "Planning study week",             done: "Built a study plan" },
};

// Subject-aware suggestion decks. The switcher picks the right one.
const SUGGESTION_DECKS = {
  maths_science: [
    { title: "I am weak in Algebra.",            hint: "What should I do next?" },
    { title: "What should I study this week?",   hint: "Balance weak topics with upcoming tests" },
    { title: "Which topic should I prioritize?", hint: "Rank by urgency and weakness" },
    { title: "I have a Maths test coming up.",   hint: "Help me prepare" },
  ],
  jee: [
    { title: "What should I focus on for JEE?",  hint: "Given my recent attempts" },
    { title: "Plan my week around my JEE test.", hint: "Respect my daily budget" },
    { title: "Which chapter should I hit first?", hint: "Rank by weakness and test urgency" },
    { title: "Give me practice on Organic Chemistry.", hint: "Start from fundamentals" },
  ],
  humanities: [
    { title: "Where am I weakest in English?",   hint: "Pick the one to fix first" },
    { title: "What should I study this week?",   hint: "Balance weak areas with upcoming tests" },
    { title: "I have an English test soon.",     hint: "Help me prepare" },
    { title: "Give me reading comprehension practice.", hint: "Short, focused drills" },
  ],
  toeic: [
    { title: "What are my weak areas in TOEIC?", hint: "Look at my recent attempts" },
    { title: "What should I study this week?",   hint: "Before my TOEIC mock test" },
    { title: "Help me on conditionals and idioms.", hint: "Where should I start?" },
    { title: "Give me a plan for today.",         hint: "60 minutes, highest-impact first" },
  ],
  generic: [
    { title: "What are my weak areas?",          hint: "Start here" },
    { title: "What should I study this week?",   hint: "Make me a plan" },
    { title: "Which topic should I prioritize?", hint: "Pick the single most important" },
    { title: "Help me prepare for my next test.", hint: "Use what's coming up" },
  ],
};

// ---------------------------------------------------------------- students

async function loadStudents() {
  try {
    const res = await fetch("/api/students");
    if (!res.ok) {
      const hint = res.status === 404
        ? "The server may be an older build — restart assistant-server."
        : `HTTP ${res.status}`;
      throw new Error(hint);
    }
    const data = await res.json();
    if (!Array.isArray(data.students) || data.students.length === 0) {
      throw new Error("No students in the DB. Delete data/db.sqlite3 and restart to re-seed.");
    }
    studentSelect.innerHTML = "";
    for (const s of data.students) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.name} (${s.source.toUpperCase()})`;
      studentSelect.appendChild(opt);
    }
    const preferred = new URLSearchParams(location.search).get("student") || data.default;
    studentSelect.value = data.students.find(s => s.id === preferred) ? preferred : data.students[0]?.id;
    currentStudentId = studentSelect.value;
    await loadStudent(currentStudentId);
  } catch (err) {
    studentCard.innerHTML = `<div class="card-loading" style="color:var(--err)">Could not load students.<br><span style="font-size:11px">${esc(err.message)}</span></div>`;
    console.error("loadStudents failed:", err);
  }
}

async function loadStudent(studentId) {
  try {
    const res = await fetch(`/api/student?id=${encodeURIComponent(studentId)}`);
    const data = await res.json();
    currentProfile = data.profile;

    const p = data.profile;
    const weakTags = (p.weak_topics || []).map(t => `<span class="tag weak">${esc(t)}</span>`).join("");
    const strongTags = (p.strong_topics || []).map(t => `<span class="tag strong">${esc(t)}</span>`).join("");
    const nextTest = (data.upcoming_tests || [])[0];
    const initials = (p.name || "?").split(/\s+/).map(w => w[0]).slice(0, 2).join("").toUpperCase();
    const gradeLine = p.grade ? `Grade ${p.grade}${p.board ? " · " + esc(p.board) : ""}` : esc(p.target_exam || p.source);

    let nextTestHtml = "";
    if (nextTest) {
      const days = daysUntil(nextTest.date);
      const whenLabel = days === 0 ? "today" : days === 1 ? "tomorrow" : `in ${days} days`;
      nextTestHtml = `
        <div class="field">
          <div class="field-label">Next test</div>
          <div class="next-test">
            ${esc(nextTest.subject || nextTest.test_name)} — <span class="when">${whenLabel}</span>
            <div style="color:var(--text-faint);font-size:11.5px;margin-top:3px;">
              ${(nextTest.topics || []).map(esc).join(" · ")}
            </div>
          </div>
        </div>`;
    }

    studentCard.innerHTML = `
      <div class="student-header">
        <div class="avatar">${esc(initials)}</div>
        <div>
          <div class="student-name">${esc(p.name)}</div>
          <div class="student-meta">${gradeLine}</div>
        </div>
      </div>
      <div class="budget-row">
        <span style="color:var(--text-faint)">Daily budget:</span> ${p.daily_study_time_minutes} min
      </div>
      <div class="field">
        <div class="field-label">Weak${(p.weak_topics || []).length ? "" : " (from attempts)"}</div>
        <div class="tag-row">${weakTags || '<span style="color:var(--text-faint);font-size:12px">Inferred dynamically</span>'}</div>
      </div>
      <div class="field">
        <div class="field-label">Strong</div>
        <div class="tag-row">${strongTags || '<span style="color:var(--text-faint);font-size:12px">Inferred dynamically</span>'}</div>
      </div>
      ${nextTestHtml}
    `;

    renderEmptyState(p);
  } catch (err) {
    studentCard.innerHTML = `<div class="card-loading" style="color:var(--err)">Could not load student.<br><span style="font-size:11px">${esc(err.message)}</span></div>`;
    console.error("loadStudent failed:", err);
  }
}

function renderEmptyState(profile) {
  emptyTitleEl.textContent = `Hi ${profile.name} — what shall we work on?`;
  const deck = pickDeck(profile);
  emptySuggestionsEl.innerHTML = deck.map(s => `
    <button class="sugg-card" data-prompt="${esc(buildPrompt(s))}">
      <div class="sugg-title">${esc(s.title)}</div>
      <div class="sugg-hint">${esc(s.hint)}</div>
    </button>
  `).join("");
  emptySuggestionsEl.querySelectorAll(".sugg-card").forEach(btn => {
    btn.addEventListener("click", () => sendMessage(btn.dataset.prompt));
  });
}

function pickDeck(profile) {
  const exam = (profile.target_exam || "").toLowerCase();
  const weak = (profile.weak_topics || []).join(" ").toLowerCase();
  if (/jee|neet/.test(exam)) return SUGGESTION_DECKS.jee;
  if (/toeic|ielts|sat|gre/.test(exam)) return SUGGESTION_DECKS.toeic;
  // Humanities-leaning students (declared weak in English / History) get a
  // different deck even if their board is CBSE.
  if (/english|history|reading comprehension|grammar/.test(weak)) return SUGGESTION_DECKS.humanities;
  const board = (profile.board || "").toLowerCase();
  if (board === "cbse" || /cbse|school/.test(exam)) return SUGGESTION_DECKS.maths_science;
  return SUGGESTION_DECKS.generic;
}

function buildPrompt(s) {
  const hint = s.hint.endsWith("?") || s.hint.endsWith(".") ? s.hint : s.hint + ".";
  return `${s.title} ${hint}`;
}

loadStudents();

// --------------------------------------------------------------- helpers

function esc(str) {
  return String(str ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function daysUntil(isoDate) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const d = new Date(isoDate + "T00:00:00");
  return Math.round((d - today) / (1000 * 60 * 60 * 24));
}

// Tiny markdown-ish renderer. Paragraphs, lists, headings, bold, italic,
// inline code, citations [M###], and KaTeX placeholders (which we splice in
// after the HTML build).
function renderMarkdown(text) {
  // Pluck math blocks first so their content isn't escaped. Replace them
  // with numeric placeholders, splice them back after escape+markdown.
  const mathBlocks = [];
  const placeholder = (i) => ` MATH${i} `;
  let withoutMath = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, expr) => {
    mathBlocks.push({ display: true, tex: expr.trim() });
    return placeholder(mathBlocks.length - 1);
  });
  withoutMath = withoutMath.replace(/\$([^$\n]+?)\$/g, (_, expr) => {
    mathBlocks.push({ display: false, tex: expr.trim() });
    return placeholder(mathBlocks.length - 1);
  });

  const escaped = esc(withoutMath);
  const lines = escaped.split("\n");
  const out = [];
  let i = 0;

  const applyInline = (s) => s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(M\d+)\]/g, (_, id) => {
      const title = materialsCache[id]?.title;
      const topic = materialsCache[id]?.topic;
      const tooltip = title ? ` title="${esc(title)}${topic ? " — " + esc(topic) : ""}"` : "";
      return `<span class="cite"${tooltip}>${id}</span>`;
    });

  while (i < lines.length) {
    const line = lines[i];

    if (/^#{1,3}\s/.test(line)) {
      const m = line.match(/^(#{1,3})\s+(.*)$/);
      const level = Math.min(m[1].length + 1, 3);
      out.push(`<h${level}>${applyInline(m[2])}</h${level}>`);
      i++; continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${applyInline(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`);
        i++;
      }
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(`<li>${applyInline(lines[i].replace(/^\s*\d+\.\s+/, ""))}</li>`);
        i++;
      }
      out.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    if (line.trim() === "") { i++; continue; }

    const paraLines = [];
    while (i < lines.length && lines[i].trim() !== "" && !/^#{1,3}\s|^\s*[-*]\s+|^\s*\d+\.\s+/.test(lines[i])) {
      paraLines.push(lines[i]);
      i++;
    }
    out.push(`<p>${applyInline(paraLines.join("<br>"))}</p>`);
  }

  let html = out.join("");

  // Splice math placeholders back, rendered by KaTeX.
  html = html.replace(/ MATH(\d+) /g, (_, idx) => {
    const blk = mathBlocks[Number(idx)];
    if (!blk || typeof katex === "undefined") return esc(blk?.tex || "");
    try {
      return katex.renderToString(blk.tex, { displayMode: blk.display, throwOnError: false });
    } catch {
      return `<code>${esc(blk.tex)}</code>`;
    }
  });

  return html;
}

function ensureMessagesContainer() {
  if (messagesInner) return messagesInner;
  if (emptyEl?.parentNode) emptyEl.remove();
  messagesInner = document.createElement("div");
  messagesInner.className = "messages-inner";
  messagesEl.appendChild(messagesInner);
  return messagesInner;
}

function addUserMessage(text) {
  const container = ensureMessagesContainer();
  const wrap = document.createElement("div");
  wrap.className = "message user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  container.appendChild(wrap);
  return wrap;
}

function addAssistantMessage() {
  const container = ensureMessagesContainer();
  const wrap = document.createElement("div");
  wrap.className = "message assistant";

  const tools = document.createElement("div");
  tools.className = "tool-status empty";
  wrap.appendChild(tools);

  const content = document.createElement("div");
  content.className = "content";
  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.textContent = "Thinking";
  content.appendChild(thinking);
  wrap.appendChild(content);

  container.appendChild(wrap);
  return { wrap, tools, content, thinking };
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// --------------------------------------------------------------- streaming

async function sendMessage(text) {
  if (streaming || !text.trim()) return;
  streaming = true;
  sendBtn.disabled = true;
  input.value = "";
  input.style.height = "auto";

  addUserMessage(text);
  const { tools, content, thinking } = addAssistantMessage();
  scrollToBottom();

  let rawText = "";
  const toolLines = new Map();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history,
        student_id: currentStudentId,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    if (!res.body) throw new Error("No response body");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const dataLine = frame.split("\n").find(l => l.startsWith("data:"));
        if (!dataLine) continue;
        handleEvent(JSON.parse(dataLine.slice(5).trim()));
      }
    }
  } catch (err) {
    content.innerHTML = `<div class="error-block">Stream failed: ${esc(err.message)}</div>`;
  } finally {
    streaming = false;
    sendBtn.disabled = false;
    input.focus();
  }

  function handleEvent(evt) {
    switch (evt.kind) {
      case "session":
        return;

      case "tool_use": {
        tools.classList.remove("empty");
        const line = document.createElement("div");
        line.className = "tool-line running";
        line.innerHTML = `<span class="dot"></span><span class="label">${esc(TOOL_VERBS[evt.name]?.running || evt.name)}…</span>`;
        tools.appendChild(line);
        toolLines.set(evt.name, line);
        scrollToBottom();
        return;
      }

      case "tool_result": {
        const line = toolLines.get(evt.name);
        if (line) {
          line.classList.remove("running");
          line.classList.add("done");
          const label = line.querySelector(".label");
          const n = evt.summary?.count;
          const suffix = n != null ? ` (${n})` : "";
          label.textContent = (TOOL_VERBS[evt.name]?.done || evt.name) + suffix;
        }
        return;
      }

      case "text": {
        if (thinking.parentNode) thinking.remove();
        rawText += evt.delta;
        content.innerHTML = renderMarkdown(rawText);
        scrollToBottom();
        return;
      }

      case "done": {
        history.push({ role: "user", content: text });
        history.push({ role: "assistant", content: rawText });
        if (evt.citations?.length) {
          for (const c of evt.citations) materialsCache[c.material_id] = c;
          content.innerHTML = renderMarkdown(rawText);
        }
        return;
      }

      case "error": {
        if (thinking.parentNode) thinking.remove();
        content.innerHTML = `<div class="error-block">${esc(evt.message)}</div>`;
        return;
      }
    }
  }
}

// --------------------------------------------------------------- form wiring

form.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(input.value);
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});

studentSelect.addEventListener("change", async () => {
  // Switching context mid-conversation would be confusing. Reset so the next
  // chat starts fresh against the new student.
  currentStudentId = studentSelect.value;
  history = [];
  materialsCache = {};
  messagesInner?.remove();
  messagesInner = null;
  messagesEl.appendChild(emptyEl);
  await loadStudent(currentStudentId);
});

resetBtn.addEventListener("click", () => {
  history = [];
  materialsCache = {};
  messagesInner?.remove();
  messagesInner = null;
  messagesEl.appendChild(emptyEl);
});

input.focus();
