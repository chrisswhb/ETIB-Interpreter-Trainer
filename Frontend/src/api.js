/**
 * ETIB API Client
 * ================
 * Centralised fetch helpers for all backend endpoints.
 * Import functions from here — don't write raw fetch() calls in main.js.
 *
 * Person 3 maintains this file.
 */

export const SERVER_BASE = import.meta.env.VITE_API_URL || 'https://joe-haddad3-etib-backend.hf.space';
const BASE = SERVER_BASE + '/api';

// ── Auth token (stored in localStorage) ──────────────────────────────────────
const AUTH_TOKEN_KEY = 'etib_auth_token';
export function getAuthToken() { return localStorage.getItem(AUTH_TOKEN_KEY) || ''; }
export function saveAuthToken(token) {
  if (token) localStorage.setItem(AUTH_TOKEN_KEY, token.trim());
  else localStorage.removeItem(AUTH_TOKEN_KEY);
}

// The platform runs on CENTRAL server-side keys — Gemini for text generation and
// evaluation, Groq for Whisper ASR. Students no longer supply a key of their own
// (21 Aug 2026), so nothing key-related is stored in the browser or sent from it.
let _currentUserId = null;
export function setCurrentUserId(id) { _currentUserId = id || null; }

// ── Body helpers (no custom headers — works with HF Spaces CORS proxy) ───────
function authBody(extra = {}) {
  const token = getAuthToken();
  return token ? { auth_token: token, ...extra } : { ...extra };
}
const groqBody = authBody;

const JSON_HEADERS = { 'Content-Type': 'application/json' };

async function safeFetch(url, options = {}) {
  try {
    return await fetch(url, options);
  } catch {
    // Free-tier hosting sleeps after ~15 min idle and takes 2-3 min to wake —
    // most "cannot reach" failures during testing were exactly this.
    throw new Error('Cannot reach the server — it may be waking up from sleep (free hosting takes 2–3 minutes). Please wait a moment and try again; also check your internet connection.');
  }
}

// Wake the backend as soon as the app opens and keep it awake during the
// session (free hosting sleeps after ~15 min idle; a sleeping server made
// transcriptions fail for testers). Fire-and-forget — errors ignored.
export function pingServer() {
  fetch(SERVER_BASE, { method: 'GET', mode: 'no-cors' }).catch(() => {});
}

async function parseJsonResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Server error (${res.status}). Please try again.`);
  return data;
}

export async function loginUser(credentials) {
  const res = await safeFetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(credentials)
  });
  return parseJsonResponse(res);
}

export async function signupUser(account) {
  const res = await safeFetch(`${BASE}/auth/signup`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(account)
  });
  return parseJsonResponse(res);
}

export async function getCurrentUser() {
  const token = getAuthToken();
  const res = await safeFetch(`${BASE}/auth/me?auth_token=${encodeURIComponent(token)}`);
  return parseJsonResponse(res);
}

export async function logoutUser() {
  const res = await safeFetch(`${BASE}/auth/logout`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(authBody())
  });
  return parseJsonResponse(res);
}

export async function generateSpeech(params) {
  const res = await safeFetch(`${BASE}/module-a/generate`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody(params))
  });
  return parseJsonResponse(res);
}

export async function generateMaterialsFromScript({ script, language, domain }) {
  const res = await safeFetch(`${BASE}/module-a/materials-from-script`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody({ script, language, domain }))
  });
  return parseJsonResponse(res);
}

export async function generateSpeechFromDocument(files, params) {
  const form = new FormData();
  const fileList = Array.isArray(files) ? files : [files];
  fileList.forEach(f => form.append('documents', f));
  form.append('auth_token', getAuthToken());
  Object.entries(params).forEach(([key, value]) => form.append(key, String(value)));
  const res = await safeFetch(`${BASE}/module-a/from-document`, { method: 'POST', body: form });
  return parseJsonResponse(res);
}

export async function retrieveDocumentContext(files, params) {
  const form = new FormData();
  const fileList = Array.isArray(files) ? files : [files];
  fileList.forEach(f => form.append('documents', f));
  form.append('auth_token', getAuthToken());
  Object.entries(params).forEach(([key, value]) => form.append(key, String(value)));
  const res = await safeFetch(`${BASE}/module-a/retrieve-document-context`, { method: 'POST', body: form });
  return parseJsonResponse(res);
}

export async function textToSpeech(params) {
  const res = await safeFetch(`${BASE}/module-b/tts`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(authBody(params))
  });
  return parseJsonResponse(res);
}

export async function transcribeAudio(audioFile, language = 'ar', sourceText = '') {
  const form = new FormData();
  form.append('audio', audioFile, audioFile.name || 'recording.webm');
  form.append('language', language);
  form.append('auth_token', getAuthToken());
  if (sourceText) form.append('source_text', sourceText);
  const res = await safeFetch(`${BASE}/module-c/transcribe`, { method: 'POST', body: form });
  return parseJsonResponse(res);
}

export async function generateMaterials(params) {
  const res = await safeFetch(`${BASE}/module-b/materials`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody(params))
  });
  return parseJsonResponse(res);
}

export async function downloadGlossary(params) {
  const res = await safeFetch(`${BASE}/module-b/glossary/download`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(authBody(params))
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return res.blob();
}

// Per-user glossary correction memory (Lina 7 Aug): a logged-in user's
// corrected terms are stored server-side and follow their account.
export async function getGlossaryCorrections() {
  const token = getAuthToken();
  if (!token) return { corrections: [] };   // guests have no correction memory at all
  const res = await safeFetch(`${BASE}/module-b/glossary/corrections?auth_token=${encodeURIComponent(token)}`);
  return parseJsonResponse(res);
}

export async function saveGlossaryCorrection(correction) {
  const token = getAuthToken();
  if (!token) return { saved: 0 };          // guests: a fix stays on screen, it is never saved
  const res = await safeFetch(`${BASE}/module-b/glossary/corrections`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(authBody({ correction })),
  });
  return parseJsonResponse(res);
}

export async function uploadGlossary(file) {
  const form = new FormData();
  form.append('glossary', file, file.name || 'glossary');
  form.append('auth_token', getAuthToken());
  const res = await safeFetch(`${BASE}/module-b/glossary/upload`, { method: 'POST', body: form });
  return parseJsonResponse(res);
}

export async function evaluatePerformance(sourceScript, transcript, language) {
  const res = await safeFetch(`${BASE}/module-d/evaluate`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody({ source_script: sourceScript, transcript, language }))
  });
  return parseJsonResponse(res);
}

export async function tashkeelCompare(sourceText, transcript, language) {
  const res = await safeFetch(`${BASE}/module-d/tashkeel-compare`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody({ source_text: sourceText, transcript, language }))
  });
  return parseJsonResponse(res);
}

export async function alignPronunciation(audioFile, segments, sourceText, language) {
  const form = new FormData();
  form.append('audio', audioFile, audioFile.name || 'recording.webm');
  form.append('segments', JSON.stringify(segments || []));
  form.append('source_text', sourceText || '');
  form.append('language', language || 'ar');
  form.append('auth_token', getAuthToken());
  const res = await safeFetch(`${BASE}/module-c/align`, { method: 'POST', body: form });
  return parseJsonResponse(res);
}

export async function getPronunciationReport(alignment, sourceText, language) {
  const res = await safeFetch(`${BASE}/module-d/pronunciation`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody({ alignment, source_text: sourceText, language }))
  });
  return parseJsonResponse(res);
}

export async function generateFeedback(params) {
  const res = await safeFetch(`${BASE}/module-d/feedback`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody(params))
  });
  return parseJsonResponse(res);
}

// ── UN Library ───────────────────────────────────────────────────────────────

export async function searchUNLibrary({ q, language, domain, limit = 10 }) {
  const params = new URLSearchParams({ language, limit });
  if (q) params.append('q', q);
  if (domain) params.append('domain', domain);
  const res = await safeFetch(`${BASE}/library/search?${params}`);
  return parseJsonResponse(res);
}

export async function fetchUNDocument({ pdf_url, web_url, un_id, title, language, description = '' }) {
  const res = await safeFetch(`${BASE}/library/fetch`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody({ pdf_url, web_url, un_id, title, language, description })),
  });
  return parseJsonResponse(res);
}

export async function getSavedSpeeches({ language, domain } = {}) {
  const params = new URLSearchParams();
  if (language) params.append('language', language);
  if (domain)   params.append('domain', domain);
  const res = await safeFetch(`${BASE}/library/saved?${params}`);
  return parseJsonResponse(res);
}

export async function getSavedSpeech(un_id) {
  const res = await safeFetch(`${BASE}/library/saved/${un_id}`);
  return parseJsonResponse(res);
}

export async function deleteSavedSpeech(un_id) {
  const res = await safeFetch(`${BASE}/library/saved/${encodeURIComponent(un_id)}`, { method: 'DELETE' });
  return parseJsonResponse(res);
}

export async function evaluateWithAudio(audioFile, sourceScript, language, sourceLanguage, domain, glossary, mode, transcriptOverride) {
  const form = new FormData();
  form.append('audio', audioFile, audioFile.name || 'recording.webm');
  form.append('source_script', sourceScript || '');
  form.append('language', language || 'ar');
  form.append('source_language', sourceLanguage || language || 'ar');
  form.append('auth_token', getAuthToken());
  if (domain) form.append('domain', domain);
  // Interpretation mode — lets the evaluator apply mode-aware pause tolerance
  // (e.g. Ear-Voice Span lag in simultaneous is not a fluency error).
  form.append('mode', mode || 'consecutive');
  // Student-reviewed glossary → terminology errors are judged against it
  if (Array.isArray(glossary) && glossary.length > 0) {
    form.append('glossary', JSON.stringify(glossary));
  }
  // Transcript the student corrected in Module C (ETIB feedback 21 Aug 2026):
  // the text is evaluated as corrected, while pauses/hesitations/rate keep
  // being measured from the real audio.
  if (transcriptOverride && transcriptOverride.trim()) {
    form.append('transcript_override', transcriptOverride.trim());
  }
  const res = await safeFetch(`${BASE}/module-d/full-evaluation`, { method: 'POST', body: form });
  return parseJsonResponse(res);
}

export async function fetchWebPage(url) {
  const res = await safeFetch(`${BASE}/module-a/fetch-url`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(authBody({ url })),
  });
  return parseJsonResponse(res);
}

export async function getSessionHistory(limit = 20) {
  const token = getAuthToken();
  const res = await safeFetch(`${BASE}/module-d/sessions?limit=${limit}&auth_token=${encodeURIComponent(token)}`);
  return parseJsonResponse(res);
}

export async function getAdaptiveParams() {
  const token = getAuthToken();
  const res = await safeFetch(`${BASE}/module-d/adaptive-params?auth_token=${encodeURIComponent(token)}`);
  return parseJsonResponse(res);
}

export async function sendChatMessage(messages) {
  const res = await safeFetch(`${BASE}/chat/message`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(groqBody({ messages })),
  });
  return parseJsonResponse(res);
}
