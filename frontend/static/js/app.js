/**
 * ETIB Frontend — app.js
 *
 * Handles:
 *   - Tab switching
 *   - Exercise loading from /api/exercises
 *   - Audio recording via MediaRecorder API
 *   - POST /api/analyze → render i'rab findings + delivery metrics
 *   - POST /api/evaluate-translation → render Layer 1 feedback
 */

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(tc => tc.style.display = "none");
    btn.classList.add("active");
    const target = btn.dataset.tab;
    document.getElementById(`tab-${target}`).style.display = "block";
  });
});

// ── State ─────────────────────────────────────────────────────────────────────
let currentExercise = null;
let mediaRecorder   = null;
let audioChunks     = [];
let recordedBlob    = null;
let timerInterval   = null;
let secondsElapsed  = 0;

// ── Load exercises ─────────────────────────────────────────────────────────────
async function loadExercises() {
  const list = document.getElementById("exercise-list");
  try {
    const resp = await fetch("/api/exercises");
    const exercises = await resp.json();

    list.innerHTML = "";
    exercises.forEach(ex => {
      const card = document.createElement("div");
      card.className = "exercise-card";
      card.innerHTML = `
        <div class="ex-id">${ex.id}</div>
        <div class="ex-title">${ex.title}</div>
        <div class="ex-desc">${ex.description}</div>
      `;
      card.addEventListener("click", () => openExercise(ex));
      list.appendChild(card);
    });
  } catch (err) {
    list.innerHTML = `<p style="color:#c0392b">خطأ في تحميل التمارين: ${err.message}</p>`;
  }
}

loadExercises();
loadGroqStatus();

async function loadGroqStatus() {
  const el = document.getElementById("groq-status");
  if (!el) return;
  try {
    const resp = await fetch("/api/groq-status");
    const data = await resp.json();
    el.textContent = data.configured
      ? "Groq مفعّل: التشكيل وWhisper scoring جاهزان"
      : "Groq غير مفعّل: أضف GROQ_API_KEY في .env لتحسين الدقة";
    el.className = `status-note ${data.configured ? "ok" : "warn"}`;
  } catch {
    el.textContent = "";
  }
}

document.getElementById("diacritize-btn").addEventListener("click", async () => {
  const textarea = document.getElementById("custom-reference");
  const text = textarea.value.trim();
  if (!text) {
    alert("أدخل نصًا عربيًا أولًا.");
    return;
  }

  const btn = document.getElementById("diacritize-btn");
  btn.disabled = true;
  btn.textContent = "جارٍ التشكيل...";
  try {
    const resp = await fetch("/api/diacritize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await resp.json();
    if (data.error) {
      alert(`تعذر التشكيل: ${data.error}`);
    }
    textarea.value = data.diacritized_text || text;
  } catch (err) {
    alert("حدث خطأ أثناء التشكيل: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "شكّل النص تلقائيًا";
    loadGroqStatus();
  }
});

document.getElementById("custom-start-btn").addEventListener("click", () => {
  const reference = document.getElementById("custom-reference").value.trim();
  if (!reference) {
    alert("أدخل جملة عربية مُشكّلة أولًا.");
    return;
  }
  openExercise({
    id: "custom",
    title: "تدريب مخصص",
    description: "تحليل نطق الحركات النهائية والتنوين في نص عربي مخصص.",
    reference,
    focus: extractFinalHarakaWords(reference),
    tip: "لأفضل نتيجة، أدخل النص بالتشكيل الكامل وخاصة الحركة الأخيرة لكل كلمة تريد تقييمها.",
  });
});

function extractFinalHarakaWords(sentence) {
  const finalHaraka = /[\u064b-\u0650]$/;
  return sentence
    .split(/\s+/)
    .map(w => w.replace(/[،,.!?؛:]+$/g, ""))
    .filter(w => finalHaraka.test(w));
}

// ── Open exercise ──────────────────────────────────────────────────────────────
function openExercise(ex) {
  currentExercise = ex;

  document.getElementById("exercise-selector").style.display = "none";
  const panel = document.getElementById("exercise-panel");
  panel.style.display = "block";

  document.getElementById("exercise-title").textContent = ex.title;
  document.getElementById("exercise-description").textContent = ex.description;
  document.getElementById("exercise-tip").textContent = ex.tip || "";

  // Render reference sentence with focus words highlighted
  renderReferenceSentence(ex.reference, ex.focus || []);

  // Reset UI state
  resetRecorder();
  document.getElementById("results-panel").style.display = "none";
}

function renderReferenceSentence(sentence, focusWords) {
  const container = document.getElementById("reference-display");
  let html = sentence;
  focusWords.forEach(fw => {
    html = html.replace(fw, `<span class="focus-word">${fw}</span>`);
  });
  container.innerHTML = html;
  container.setAttribute("dir", "rtl");
}

// ── Back button ────────────────────────────────────────────────────────────────
document.getElementById("back-btn").addEventListener("click", () => {
  document.getElementById("exercise-panel").style.display = "none";
  document.getElementById("exercise-selector").style.display = "block";
  resetRecorder();
});

// ── Recording ──────────────────────────────────────────────────────────────────
const recordBtn   = document.getElementById("record-btn");
const recordLabel = document.getElementById("record-label");
const timerEl     = document.getElementById("timer");
const playback    = document.getElementById("playback");
const analyzeBtn  = document.getElementById("analyze-btn");

recordBtn.addEventListener("click", toggleRecording);

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    // Stop
    mediaRecorder.stop();
    clearInterval(timerInterval);
    recordBtn.classList.remove("active");
    recordLabel.textContent = "إعادة التسجيل";
    timerEl.style.display = "none";
  } else {
    // Start
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      recordedBlob = null;
      analyzeBtn.style.display = "none";
      playback.style.display = "none";
      document.getElementById("results-panel").style.display = "none";

      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        recordedBlob = new Blob(audioChunks, { type: "audio/webm" });
        const url = URL.createObjectURL(recordedBlob);
        playback.src = url;
        playback.style.display = "block";
        analyzeBtn.style.display = "inline-block";
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.start(100);
      recordBtn.classList.add("active");
      recordLabel.textContent = "إيقاف التسجيل";
      secondsElapsed = 0;
      timerEl.style.display = "block";
      timerEl.textContent = "00:00";
      timerInterval = setInterval(() => {
        secondsElapsed++;
        const m = String(Math.floor(secondsElapsed / 60)).padStart(2, "0");
        const s = String(secondsElapsed % 60).padStart(2, "0");
        timerEl.textContent = `${m}:${s}`;
      }, 1000);
    } catch (err) {
      alert("لا يمكن الوصول إلى الميكروفون: " + err.message);
    }
  }
}

function resetRecorder() {
  if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
  clearInterval(timerInterval);
  recordBtn.classList.remove("active");
  recordLabel.textContent = "ابدأ التسجيل";
  timerEl.style.display = "none";
  playback.style.display = "none";
  analyzeBtn.style.display = "none";
  audioChunks = [];
  recordedBlob = null;
}

// ── Analyze ────────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", analyzeRecording);

async function analyzeRecording() {
  if (!recordedBlob || !currentExercise) return;

  const spinner    = document.getElementById("analyzing-spinner");
  const resultsDiv = document.getElementById("results-panel");

  analyzeBtn.style.display = "none";
  spinner.style.display    = "block";
  resultsDiv.style.display = "none";

  const formData = new FormData();
  formData.append("audio", recordedBlob, "recording.webm");
  formData.append("reference_sentence", currentExercise.reference);
  formData.append("focus_words", JSON.stringify(currentExercise.focus || []));
  formData.append("exercise_id", currentExercise.id);
  formData.append("feedback_language", "ar");

  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: formData });
    if (!resp.ok) throw new Error(`Server error ${resp.status}`);
    const data = await resp.json();

    renderResults(data);
  } catch (err) {
    alert("حدث خطأ أثناء التحليل: " + err.message);
  } finally {
    spinner.style.display = "none";
    analyzeBtn.style.display = "inline-block";
  }
}

// ── Render results ─────────────────────────────────────────────────────────────
function renderResults(data) {
  const panel = document.getElementById("results-panel");
  panel.style.display = "block";

  // Word findings
  const findingsDiv = document.getElementById("word-findings");
  findingsDiv.innerHTML = "";

  (data.iraab_findings || []).forEach(f => {
    const statusLabel = { correct: "صحيح ✓", incorrect: "خطأ ✗", uncertain: "غير محدد ?" }[f.status] || f.status;
    const badgeClass  = `badge-${f.status}`;
    const cardClass   = f.status;

    const card = document.createElement("div");
    card.className = `finding-card ${cardClass}`;

    const confPct = Math.round((f.detection_confidence || 0) * 100);
    const notesHtml = (f.pronunciation_notes || []).length
      ? `<ul class="pronunciation-notes">${f.pronunciation_notes.map(n => `<li>${n}</li>`).join("")}</ul>`
      : "";
    const endingLabels = {
      fatha: "فتحة (a)", damma: "ضمة (u)", kasra: "كسرة (i)",
      tanwin_fath: "تنوين فتح (an)", tanwin_damm: "تنوين ضم (un)", tanwin_kasr: "تنوين كسر (in)",
      none: "—", uncertain: "غير محدد",
    };

    card.innerHTML = `
      <div class="finding-header">
        <span class="finding-word">${f.word}</span>
        <span class="badge ${badgeClass}">${statusLabel}</span>
      </div>
      <div class="ending-row">
        <span>المطلوب: <strong>${endingLabels[f.expected_ending] || f.expected_ending}</strong></span>
        <span>المكتشف: <strong>${endingLabels[f.detected_ending] || f.detected_ending}</strong></span>
      </div>
      <div class="confidence-bar-wrap">
        ثقة الكشف: ${confPct}%
        <div class="confidence-bar">
          <div class="confidence-fill" style="width:${confPct}%"></div>
        </div>
      </div>
      ${f.explanation ? `<div class="finding-explanation">${f.explanation}</div>` : ""}
      ${notesHtml}
    `;
    findingsDiv.appendChild(card);
  });

  // Delivery metrics
  const delivery = data.delivery || {};
  const metricsDiv = document.getElementById("delivery-metrics");
  const durationS = delivery.duration_ms ? (delivery.duration_ms / 1000).toFixed(1) : "—";

  metricsDiv.innerHTML = `
    <div class="delivery-metrics-grid">
      <div class="metric-item">
        <div class="metric-value">${durationS}s</div>
        <div class="metric-label">مدة التسجيل</div>
      </div>
      <div class="metric-item">
        <div class="metric-value">${delivery.speech_rate_wpm ?? "—"}</div>
        <div class="metric-label">كلمة/دقيقة</div>
      </div>
      <div class="metric-item">
        <div class="metric-value">${(delivery.pauses || []).length}</div>
        <div class="metric-label">توقفات</div>
      </div>
      <div class="metric-item">
        <div class="metric-value">${delivery.long_pause_count ?? 0}</div>
        <div class="metric-label">توقفات طويلة</div>
      </div>
      <div class="metric-item">
        <div class="metric-value">${(delivery.repetitions || []).length}</div>
        <div class="metric-label">تكرارات</div>
      </div>
      <div class="metric-item">
        <div class="metric-value">${(delivery.filled_pauses || []).length}</div>
        <div class="metric-label">توقفات ملء</div>
      </div>
    </div>
  `;

  // Debug JSON
  document.getElementById("debug-json").textContent = JSON.stringify(data, null, 2);

  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Translation evaluation ─────────────────────────────────────────────────────
document.getElementById("evaluate-btn").addEventListener("click", async () => {
  const sourceText    = document.getElementById("source-text").value.trim();
  const studentTrans  = document.getElementById("student-translation").value.trim();

  if (!sourceText || !studentTrans) {
    alert("يرجى إدخال النص المصدر والترجمة.");
    return;
  }

  const spinner    = document.getElementById("eval-spinner");
  const resultsDiv = document.getElementById("eval-results");
  const evalBtn    = document.getElementById("evaluate-btn");

  evalBtn.disabled     = true;
  spinner.style.display = "block";
  resultsDiv.style.display = "none";

  const payload = {
    source_language:      document.getElementById("source-lang").value,
    target_language:      document.getElementById("target-lang").value,
    feedback_language:    document.getElementById("feedback-lang").value,
    source_text:          sourceText,
    student_translation:  studentTrans,
    reference_materials:  [],
    known_interference_matches: [],
  };

  try {
    const resp = await fetch("/api/evaluate-translation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`Server error ${resp.status}`);
    const data = await resp.json();
    renderEvalResults(data);
  } catch (err) {
    alert("حدث خطأ: " + err.message);
  } finally {
    spinner.style.display = "none";
    evalBtn.disabled = false;
  }
});

function renderEvalResults(data) {
  const resultsDiv = document.getElementById("eval-results");
  resultsDiv.style.display = "block";

  // Overall summary
  document.getElementById("eval-summary").innerHTML =
    `<strong>الملخص العام:</strong> ${data.overall_summary || "—"}`;

  // Segments
  const segsDiv = document.getElementById("eval-segments");
  segsDiv.innerHTML = "";

  (data.segments || []).forEach((seg, i) => {
    const block = document.createElement("div");
    block.className = "segment-block";

    const findingsHtml = (seg.findings || []).map(f => `
      <div class="finding-item ${f.severity}">
        <span class="finding-dim">${f.dimension}</span> ·
        <span class="finding-span">"${f.target_span}"</span>
        — ${f.explanation || ""}
        ${f.correction ? `<br><em>الاقتراح: ${f.correction}</em>` : ""}
      </div>
    `).join("") || "<p style='color:var(--muted);font-size:.85rem'>لا توجد ملاحظات لهذه الفقرة ✓</p>";

    block.innerHTML = `
      <div class="segment-header">فقرة ${i + 1} — ${seg.segment_id || ""}</div>
      <div class="segment-body">
        <div class="segment-texts">
          <div class="seg-src">${seg.source_segment || ""}</div>
          <div class="seg-tgt">${seg.target_segment || ""}</div>
        </div>
        ${findingsHtml}
      </div>
    `;
    segsDiv.appendChild(block);
  });

  // Dimension summary
  const dimDiv = document.getElementById("dimension-summary");
  const dims   = data.dimension_summary || {};
  const dimNames = {
    fidelity: "الأمانة", terminology: "المصطلحات", interference: "التداخل",
    grammar: "النحو", orthography: "الإملاء", register: "الأسلوب",
  };

  dimDiv.innerHTML = Object.entries(dimNames).map(([key, label]) => `
    <div class="dim-cell">
      <div class="dim-name">${label}</div>
      <div class="dim-score">${dims[key] ?? "—"}</div>
    </div>
  `).join("");

  resultsDiv.scrollIntoView({ behavior: "smooth" });
}
