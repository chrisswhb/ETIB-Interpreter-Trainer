let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let timerInterval = null;
let secondsElapsed = 0;

const EMPTY = "-";
const recordBtn = document.getElementById("record-btn");
const recordLabel = document.getElementById("record-label");
const timerEl = document.getElementById("timer");
const playback = document.getElementById("playback");
const analyzeBtn = document.getElementById("analyze-btn");
const spinner = document.getElementById("analyzing-spinner");
const resultsPanel = document.getElementById("results-panel");
const serviceDot = document.getElementById("service-dot");
const serviceStatus = document.getElementById("service-status");

recordBtn.addEventListener("click", toggleRecording);
analyzeBtn.addEventListener("click", analyzeRecording);
checkService();

async function checkService() {
  try {
    const resp = await fetch("/api/arabic-wrapper/health");
    const data = await resp.json();
    serviceDot.className = "service-dot ok";
    serviceStatus.textContent = data.configured?.cohere
      ? "Service ready - Cohere enabled"
      : "Service online, but Cohere key is missing";
  } catch (err) {
    serviceDot.className = "service-dot bad";
    serviceStatus.textContent = "Service unavailable";
  }
}

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    clearInterval(timerInterval);
    recordBtn.classList.remove("active");
    recordLabel.textContent = "Record again";
    timerEl.style.display = "none";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    recordedBlob = null;
    analyzeBtn.style.display = "none";
    playback.style.display = "none";
    resultsPanel.style.display = "none";

    const mimeCandidates = ["audio/webm;codecs=opus", "audio/webm"];
    const selectedMime = mimeCandidates.find(type => MediaRecorder.isTypeSupported(type)) || "";
    mediaRecorder = selectedMime
      ? new MediaRecorder(stream, { mimeType: selectedMime })
      : new MediaRecorder(stream);

    mediaRecorder.ondataavailable = event => {
      if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
      const recordedType = selectedMime || mediaRecorder.mimeType || "audio/webm";
      recordedBlob = new Blob(audioChunks, { type: recordedType });
      playback.src = URL.createObjectURL(recordedBlob);
      playback.style.display = "block";
      analyzeBtn.style.display = "inline-block";
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.start(100);
    recordBtn.classList.add("active");
    recordLabel.textContent = "Stop recording";
    secondsElapsed = 0;
    timerEl.style.display = "block";
    timerEl.textContent = "00:00";
    timerInterval = setInterval(updateTimer, 1000);
  } catch (err) {
    alert("Microphone access failed: " + err.message);
  }
}

function updateTimer() {
  secondsElapsed += 1;
  const minutes = String(Math.floor(secondsElapsed / 60)).padStart(2, "0");
  const seconds = String(secondsElapsed % 60).padStart(2, "0");
  timerEl.textContent = `${minutes}:${seconds}`;
}

async function analyzeRecording() {
  if (!recordedBlob) return;

  analyzeBtn.style.display = "none";
  spinner.style.display = "block";
  resultsPanel.style.display = "none";

  const formData = new FormData();
  formData.append("audio", recordedBlob, "arabic-recording.webm");

  try {
    const resp = await fetch("/api/arabic-wrapper/transcribe-harakat", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      let detail = `Server error ${resp.status}`;
      try {
        const errData = await resp.json();
        detail = errData.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    const data = await resp.json();
    renderResult(data);
  } catch (err) {
    alert("Analysis failed: " + err.message);
  } finally {
    spinner.style.display = "none";
    analyzeBtn.style.display = "inline-block";
  }
}

function renderResult(data) {
  const transcript = data.transcript || {};
  setArabicText("raw-transcript", transcript.raw || EMPTY);
  setArabicText("diacritized-transcript", transcript.spoken_harakat || transcript.diacritized || transcript.raw || EMPTY);

  document.getElementById("provider-value").textContent = transcript.provider || EMPTY;
  document.getElementById("model-value").textContent = transcript.model || EMPTY;

  const notesBlock = document.getElementById("notes-block");
  const notesList = document.getElementById("tanween-notes");
  const spoken = data.spoken_endings || {};
  const notes = buildEndingNotes(spoken.findings || [], transcript.tanween_notes || [], spoken.error);
  notesList.innerHTML = "";
  if (notes.length) {
    notes.forEach(note => {
      const item = document.createElement("li");
      item.textContent = typeof note === "string" ? note : JSON.stringify(note);
      notesList.appendChild(item);
    });
    notesBlock.style.display = "block";
  } else {
    notesBlock.style.display = "none";
  }

  document.getElementById("debug-json").textContent = JSON.stringify(data, null, 2);
  resultsPanel.style.display = "block";
}

function buildEndingNotes(findings, tanweenNotes, acousticError) {
  const notes = [];
  if (acousticError) notes.push("Acoustic ending detector failed: " + acousticError);
  findings.slice(0, 40).forEach(item => {
    const word = item.word || EMPTY;
    const ending = labelEnding(item.detected_ending || "none");
    const confidence = Math.round((Number(item.detection_confidence) || 0) * 100);
    notes.push(`${word}: ${ending} (${confidence}%)`);
  });
  if (!notes.length && tanweenNotes.length) return tanweenNotes;
  return notes;
}

function labelEnding(label) {
  return {
    fatha: "fatha",
    damma: "damma",
    kasra: "kasra",
    tanwin_fath: "tanween fatha",
    tanwin_damm: "tanween damma",
    tanwin_kasr: "tanween kasra",
    none: "no final mark detected"
  }[label] || label;
}

function setArabicText(id, value) {
  const el = document.getElementById(id);
  el.textContent = value;
  el.classList.toggle("empty", value === EMPTY);
}