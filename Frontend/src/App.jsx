import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  generateSpeech,
  generateSpeechFromDocument,
  generateMaterialsFromScript,
  loginUser,
  logoutUser,
  signupUser,
  textToSpeech,
  downloadGlossary,
  uploadGlossary,
  getGlossaryCorrections,
  saveGlossaryCorrection,
  transcribeAudio,
  generateFeedback,
  evaluateWithAudio,
  tashkeelCompare,
  alignPronunciation,
  getPronunciationReport,
  searchUNLibrary,
  fetchUNDocument,
  getSavedSpeeches,
  getSavedSpeech,
  deleteSavedSpeech,
  getSessionHistory,
  getAdaptiveParams,
  pingServer,
  setCurrentUserId,
  saveAuthToken,
  sendChatMessage,
  fetchWebPage,
  SERVER_BASE,
} from './api.js';

// Map a raw backend/Groq error to a clear, actionable message. An invalid or
// expired Groq key surfaces as a 401 "Invalid API Key" — tell the student to
// fix it in Settings instead of showing raw JSON.
function friendlyLlmError(message, labels) {
  const m = String(message || '');
  if (/invalid[_\s-]?api[_\s-]?key|invalid api key|\b401\b|authentication|unauthorized/i.test(m)) {
    return labels.keyInvalid || 'The AI service rejected the request (authentication). Please contact the administrator.';
  }
  if (/rate.?limit|quota|\b429\b|tokens per|tpd|tpm/i.test(m)) {
    return labels.llmQuotaError || m;
  }
  if (/source_and_target_must_differ/i.test(m)) {
    return labels.sameLangError;
  }
  if (/unreadable_media|invalid data found/i.test(m)) {
    return labels.mediaUnreadable || m;
  }
  return m;
}

const UI = {
  en: {
    uiLanguage: 'Interface language',
    loginTitle: 'Sign in to ETIB Trainer',
    loginSubtitle: 'Access your interpreter training workspace.',
    signupTitle: 'Create your ETIB Trainer account',
    signupSubtitle: 'Set up an account to start interpreter training.',
    name: 'Full name',
    email: 'Email',
    password: 'Password',
    showPassword: 'Show',
    hidePassword: 'Hide',
    role: 'Role',
    loginSubmit: 'Sign in',
    loginLoading: 'Signing in...',
    signupSubmit: 'Create account',
    signupLoading: 'Creating account...',
    loginNote: 'Use your account credentials, or create a new account.',
    switchToSignup: 'Create an account',
    switchToLogin: 'Already have an account? Sign in',
    student: 'Student',
    guestName: 'Guest',
    instructor: 'Instructor',
    coordinator: 'Coordinator',
    signOut: 'Sign out',
    workspaceKicker: 'Training modules',
    workspaceTitle: 'Generate speeches, prepare materials, record interpretations, and review feedback.',
    navA: 'Generate speech',
    navB: 'Audio & materials',
    navC: 'Record & transcribe',
    navD: 'Evaluation',
    navE: 'Progress',
    progressTitle: 'My Progress',
    progressSubtitle: 'Score history and adaptive recommendations',
    progressNoSessions: 'No sessions yet — complete a full evaluation to start tracking your progress.',
    progressSessions: 'Session history',
    progressAdaptive: 'Adaptive recommendations',
    progressAdaptiveSubtitle: 'Based on your recent sessions, the platform recommends:',
    progressApply: 'Apply these settings',
    progressOverall: 'Overall',
    progressFluency: 'Fluency',
    progressCoverage: 'Coverage',
    progressErrors: 'Errors',
    progressLang: 'Language',
    progressDate: 'Date',
    progressScore: 'Score',
    progressLoading: 'Loading history…',
    progressAvgOver: 'Avg. overall',
    progressAvgFluency: 'Avg. fluency',
    progressAvgNumbers: 'Avg. number errors',
    progressTrend: 'Score trend (last 10 sessions)',
    progressLatest: 'Latest session',
    progressProblems: 'Focus areas — what to work on',
    progressImproving: 'Improving',
    progressDeclining: 'Declining',
    progressStable: 'Stable',
    progressStrengths: 'Strengths',
    progressRecs: 'Recommendations',
    severityHigh: 'HIGH', severityMedium: 'MED', severityLow: 'LOW',
    problemLabel_translation: 'Translation accuracy',
    problemLabel_numbers: 'Numbers & figures',
    problemLabel_missing: 'Missing content',
    problemLabel_silences: 'Long pauses',
    problemLabel_repetitions: 'Repetitions & self-corrections',
    problemLabel_fluency: 'Delivery fluency',
    problemLabel_coverage: 'Content coverage',
    problemTip_translation: 'Study terminology lists and practice active listening for key concepts.',
    problemTip_numbers: 'Practice shadowing speeches with high number density.',
    problemTip_missing: 'Improve note-taking and work on shorter speeches first.',
    problemTip_silences: 'Practice décalage exercises to keep output flowing.',
    problemTip_repetitions: 'Work on reformulation — commit to a rendering instead of backtracking.',
    problemTip_fluency: 'Shadow native speakers and record yourself daily.',
    problemTip_coverage: 'Focus on conveying main ideas before worrying about style.',
    problemDetail_translation: 'Wrong equivalents in {count}/{total} sessions',
    problemDetail_numbers: 'Number errors in {count}/{total} sessions',
    problemDetail_missing: 'Key ideas omitted in {count}/{total} sessions',
    problemDetail_silences: 'Excessive pauses in {count}/{total} sessions',
    problemDetail_repetitions: 'High repetitions in {count}/{total} sessions',
    problemDetail_fluency: 'Low fluency in {count}/{total} sessions',
    problemDetail_coverage: 'Low coverage in {count}/{total} sessions',
    progressTopErrors: 'Specific errors',
    progressNoData: 'Not enough data yet',
    moduleATitle: 'Generate training speech',
    groundedSourceLabel: 'Grounded in a real source:',
    ungroundedNote: 'No matching UN document was found — this speech was generated without source grounding.',
    language: 'Speech language',
    targetLanguage: 'Target language',
    topic: 'Topic',
    domain: 'Domain',
    wordCount: 'Speech length',
    wordRangeShort: 'Short',
    wordRangeMedium: 'Medium',
    wordRangeLong: 'Long',
    difficulty: 'Difficulty',
    structure: 'Discourse structure',
    mode: 'Interpretation mode',
    numbers: 'Number density',
    hesitationsSim: 'Simulate hesitations',
    pressureMode: 'Pressure mode',
    speedPressure: 'Speed pressure',
    topicShifts: 'Topic shifts',
    contextNoise: 'Context noise',
    cognitiveLoad: 'Cognitive load',
    summary: 'Summary',
    mcq: 'MCQ',
    glossary: 'Glossary',
    documentGrounding: 'Document grounding',
    documentFile: 'Source document',
    sharedSettings: 'Generation settings',
    sharedSettingsHint: 'These settings apply to both topic-only generation and document-grounded generation.',
    generationMethods: 'Choose generation source',
    topicOnlyGeneration: 'Topic-only generation',
    topicOnlyHint: 'Generate from the topic and selected settings.',
    documentGenerationHint: 'Upload a source file and generate a speech grounded in its content.',
    generateFromDocument: 'Generate from document',
    generationEmpty: 'No speech text was generated (the server may be waking up, or the request was rate-limited). Nothing was kept — please try again.',
    retrieveContext: 'Preview retrieved context',
    retrievedContext: 'Retrieved context',
    submit: 'Generate speech',
    generating: 'Generating - please wait...',
    wordsUnit: 'words',
    moduleBTitle: 'Audio & Pedagogical Materials',
    moduleBBody: 'Will produce: audio playback, key terms, glossary, and MCQ.',
    moduleCTitle: 'Record & Transcribe',
    moduleCBody: 'Will allow: in-browser recording, ASR transcription, timestamped transcript, and playback.',
    moduleDTitle: 'Performance Evaluation',
    moduleDBody: 'Will detect: hesitations, omissions, number errors, terminology issues, and recurrent weaknesses.',
    comingSoon: 'Coming soon',
    errorPrefix: 'Error',
    noSpeechYet: 'Generate a speech in Module A first, then come back here.',
    materialsMissing: 'The comprehension questions and glossary for this speech are missing (they can drop out on very long speeches). Generate them now:',
    materialsGenerating: 'Generating…',
    voiceAccent: 'Voice accent',
    speechRate: 'Speech rate',
    sessionRestored: 'Your previous session was restored (speech, materials, and glossary edits). You can continue where you left off.',
    sessionKeep: 'Continue',
    sessionDiscard: 'Start fresh',
    measuredRate: 'Measured pace',
    wordsPerMinute: 'words/min',
    rateTooFastHint: 'above the 100–120 wpm range recommended for simultaneous — lower the speech rate slider and regenerate',
    mediaUnreadable: 'This file could not be read. Supported formats: MP3, WAV, M4A, OGG, WebM, MP4. If it plays on your device, convert it to MP3 or WAV and retry.',
    generateAudio: 'Generate audio',
    generatingAudio: 'Generating audio...',
    materialsTitle: 'Pedagogical materials',
    generateMaterials: 'Generate materials',
    generatingMaterials: 'Generating materials...',
    materialsFailed: 'The speech was transcribed, but the materials (summary, MCQ, glossary) could not be generated — the server may be waking up (2–3 min). Click "Upload a speech" again to retry.',
    keyTerms: 'Key terms',
    summaryTitle: 'Thematic summary',
    mcqTitle: 'Comprehension questions (MCQ)',
    mcqQuestionHeader: 'Question',
    mcqOptionsHeader: 'Options',
    mcqAnswerHeader: 'Answer',
    glossaryTitle: 'Trilingual glossary (AR – FR – EN)',
    glossaryDuringRecording: 'Glossary (keep terminology visible while interpreting)',
    glossaryTermHeader: 'Term',
    glossaryArabicHeader: 'Arabic',
    glossaryFrenchHeader: 'French',
    glossaryEnglishHeader: 'English',
    glossaryDefinitionHeader: 'Definition',
    downloadGlossary: 'Download glossary (DOCX)',
    uploadGlossary: 'Import glossary',
    glossaryUploading: 'Importing…',
    glossaryUploadEmpty: 'No terms could be read from that file. Use columns like Term / Arabic / French / English (CSV, TSV, TXT, DOCX, or JSON).',
    uploadAudio: 'Upload your interpretation (MP3, WAV, M4A)',
    transcribeBtn: 'Transcribe',
    transcribing: 'Transcribing — may take 1–2 min...',
    transcriptLabel: 'Transcript',
    lowConfidence: 'Words the recognizer heard unclearly (check these)',
    arabicTip: 'For best results: speak clearly in Modern Standard Arabic (فصحى) in a quiet room.',
    noTranscriptYet: 'Record your interpretation in "Record & transcribe" first, then come back here.',
    runEvaluation: 'Run full evaluation',
    runningEvaluation: 'Analyzing your interpretation...',
    overallScore: 'Overall score',
    evalSummary: 'Overall assessment',
    strengths: 'Strengths',
    recommendations: 'Recommendations',
    algorithmicTitle: 'Automatic detections',
    longSilences: 'Long silences',
    repetitions: 'Repetitions',
    hesitations: 'Hesitations',
    filledPause: 'filled pause (euh…)',
    cutWord: 'cut word',
    numberErrors: 'Number errors',
    llmTitle: 'AI language analysis',
    languageErrors: 'Language errors',
    autoCorrections: 'Auto-corrections',
    falseStarts: 'False starts',
    lapsusLinguae: 'Lapsus linguae',
    terminologyProblems: 'Terminology problems',
    informationLoss: 'Information loss',
    noIssues: 'None detected',
    correction: 'Correction',
    source: 'Source',
    used: 'Used',
    correct: 'Correct term',
    coverageTitle: 'Content coverage',
    coverageHint: 'How much of the source speech was conveyed',
    translationErrors: 'Translation errors',
    meaningCheckTitle: 'Meaning check',
    meaningCheckNote: 'Possible changes of meaning — shown to help you learn; they do NOT affect your score. The same idea can be phrased differently, so only genuine changes of meaning are listed.',
    sourceSaid: 'Source said',
    studentSaid: 'Student said',
    correctTranslation: 'Correct',
    missingContent: 'Missing content',
    scoreScopeNote: 'This score reflects terminology, target-language accuracy, numbers/names, and delivery (pauses, décalage, fluency). Meaning/coverage below is shown for your information and does not affect the score at this stage.',
    aiDisclaimerTitle: '⚠️ AI disclaimer',
    aiDisclaimerShort: 'AI can be wrong — check the generated text, audio, transcript and scores before you rely on them.',
    aiDisclaimerMore: 'Details',
    aiDisclaimerLess: 'Close',
    aiDisclaimerGen: 'Generated speeches: The text is produced by a large language model. It can contain invented figures, non-existent report or resolution names, factual mistakes, and cultural or political bias. Never treat it as a source of facts — it is training material. In Arabic, number agreement (العدد وأحكامه), diacritic signs (التحريك) and dates are the most error-prone parts.',
    aiDisclaimerTts: 'Synthetic audio (TTS): The voice can mispronounce names, acronyms, dates and figures — especially in Arabic. What you hear does not always match the written text. Edit the text before generating the audio if a passage sounds wrong.',
    aiDisclaimerStt: 'Transcription (STT): Your recording is transcribed automatically. The transcript is an estimate, not a certain record of what you said. Arabic, code-switching and background noise lower its accuracy noticeably. Correct the transcript before running the evaluation.',
    aiDisclaimerEval: 'Evaluation: The report is an automatic training aid — not a grade, and not a substitute for your instructor. It assesses only the categories listed below; any other observations or statements are for indicative purposes only.',
    aiDisclaimerAdvice: 'What we ask of you: Verify terminology using the official databases, which serve as reliable reference sources; check figures and names against the source; correct the transcript; and report anything that looks wrong to your instructor.',
    aiDisclaimerAck: 'I understand',
    ownTextBtn: '\u270d\ufe0f Use my own text (no AI)',
    ownTextTitle: '\u270d\ufe0f My own text \u2014 audio only, no AI generation',
    ownTextHint: 'Paste a text you already have. It is used EXACTLY as written: no AI rewriting, no AI generation. Then go to \u201cAudio & materials\u201d to have it read aloud, and interpret it as usual.',
    ownTextPlaceholder: 'Paste your text here\u2026',
    ownTextUse: 'Use this text as-is',
    ownTextEmpty: 'Paste a text first.',
    ownTextBadge: 'Your own text \u2014 not AI-generated',
    ownTextMaterialsHint: 'This text was added as-is, so it has no summary, MCQ or glossary yet. Generate them with the button if you want them \u2014 the text itself is never changed.',
    scriptTitle: 'Speech text',
    editScriptBtn: '\u270e Edit the text',
    editScriptDone: '\u2713 Done',
    editScriptHint: 'Correct the text BEFORE generating the audio \u2014 mistakes, awkward figures, wrong names. The audio, your interpretation and the evaluation all use this edited version.',
    scriptRestore: '\u21ba Restore the original',
    editTranscriptBtn: '\u270e Correct the transcript',
    editTranscriptDone: '\u2713 Save the correction',
    editTranscriptHint: 'The transcript is what the speech recognizer heard. Correct any recognition error BEFORE the evaluation, so a machine mistake is not counted as yours. Pauses, hesitations and speech rate keep being measured from your real recording.',
    transcriptEditedBadge: 'transcript corrected by you',
    micWarmingUp: 'Preparing the microphone\u2026',
    micReadyHint: 'Wait for \u201cRecording\u201d before you start speaking.',
    mediaAsrWarning: 'This text comes from the automatic transcription of your file. Its accuracy depends on the audio quality (and is lower in Arabic): re-read it and correct it in \u201cAudio & materials\u201d before using it as a source.',
    fluencyScore: 'Fluency score',
    pauseAnalysisTitle: 'Pauses & flow',
    repetitionAnalysisTitle: 'Repetitions — analysis',
    numberAccuracyTitle: 'Numbers & dates',
    pronunciationAssessmentTitle: 'Pronunciation feedback',
    numCorrect: 'Correct',
    numIncorrect: 'Incorrect',
    expectedLabel: 'Should be',
    studentSaidShort: 'Said',
    pronunciationTitle: 'Pronunciation Assessment',
    pronunciationMismatches: 'Word-level clarity mismatches',
    runPronunciation: 'Run pronunciation check',
    runningPronunciation: 'Aligning audio against source — may take 1 min...',
    pronunciationScore: 'Detection certainty',
    wordView: 'Word-by-word view',
    legend: 'Legend',
    legendGood: '≥80% clear',
    legendWarn: '65–80%',
    legendPoor: '<65% — likely error',
    uncertainWords: 'Key words that sounded unclear',
    expectedForm: 'Expected',
    likelyError: 'Likely error',
    grammaticalRole: 'Grammatical role',
    noUncertain: 'No key word sounded unclear',
    whisperxBadge: 'WhisperX aligned',
    fallbackBadge: 'Whisper scores',
    needsArabic: 'Pronunciation check is available for Arabic interpretations only.',
    needsSource: 'Generate a speech in Module A first so the source text is available.',
    sourceAudio: 'Source speech audio',
    playbackSpeed: 'Playback speed',
    yourInterpretation: 'Your interpretation',
    recordBtn: '🎙 Start recording',
    stopBtn: '⏹ Stop recording',
    recordingLive: 'Recording',
    autoTranscribe: 'Transcribe automatically after recording',
    orUpload: 'Or upload an audio file',
    deleteAudio: 'Delete audio',
    redoTranscript: 'Redo transcription',
    copyText: 'Copy',
    bubbleGood: 'Good transcription',
    bubblePoor: 'Poor transcription',
    deleteTranscript: 'Delete transcript',
    closeLabel: 'Close',
    copied: 'Copied!',
    topicPlaceholder: 'Enter a topic or paste text to generate a speech…',
    namePlaceholder: 'Your full name',
    librarySearchPlaceholder: 'e.g. climate change, AI governance…',
    removeSource: 'Remove source',
    addSourceFirst: 'Add at least one source document first.',
    moreSettings: 'More ⚙',
    lessSettings: 'Less ⚙',
    optConsecutive: 'Consecutive',
    optSimultaneous: 'Simultaneous',
    optSight: 'Sight translation',
    optWellOrganized: 'Well organized',
    optDisorganized: 'Disorganized',
    optLowNumbers: 'Low numbers',
    optHighNumbers: 'High numbers',
    optNormalSpeed: 'Normal speed',
    optFast: 'Fast',
    optVeryFast: 'Very fast',
    optNoShifts: 'No topic shifts',
    optShifts: 'Topic shifts',
    domPolitics: 'Politics',
    domDiplomacy: 'Diplomacy',
    domEconomics: 'Economics',
    domHealth: 'Health',
    domEducation: 'Education',
    domClimate: 'Climate & Environment',
    domHumanRights: 'Human Rights',
    domTechnology: 'Technology & AI',
    domMigration: 'Migration & Refugees',
    domDisarmament: 'Disarmament',
    domWomen: 'Women & Gender',
    domFood: 'Food & Hunger',
    domLegal: 'Legal',
    domMedical: 'Medical',
    diffBeginner: 'Beginner',
    diffIntermediate: 'Intermediate',
    diffAdvanced: 'Advanced',
    diffHint: 'Beginner: simple vocabulary, short sentences, 1–2 numbers, slow pace. Intermediate: moderate terminology, several statistics, mixed sentences. Advanced: dense terminology, frequent numbers and names, complex syntax, fast pace.',
    wordRangeShortFull: 'Short — 120–180 words (≈ 1 min)',
    wordRangeMediumFull: 'Medium — 220–320 words (≈ 1.5–2.5 min)',
    wordRangeLongFull: 'Long — 400–550 words (≈ 3–4 min)',
    wordRangeExtendedFull: 'Extended — 650–800 words (≈ 4.5–6 min)',
    wordRangeVeryLongFull: 'Very long — 1100–1400 words (≈ 8–11 min)',
    wordRangeExtraLongFull: 'Extra long — 1500–2200 words (≈ 12–18 min)',
    wordRangeMarathonFull: 'Marathon — 2900–3600 words (≈ 24–30 min)',
    optSemiStructured: 'Semi-structured',
    optDisorganizedFull: 'Deliberately disorganized',
    scenarioLabel: 'Speaker style / setting',
    scenUNGA: 'UN General Assembly',
    scenEUParl: 'EU Parliament',
    scenArabLeague: 'Arab League summit',
    scenPress: 'Press conference',
    scenDiplomatic: 'Diplomatic meeting',
    scenDebate: 'Political debate',
    scenInterview: 'Interview',
    scenGroupGeneral: 'General / conference',
    scenGroupSimul: 'Simultaneous-oriented',
    scenGroupConsec: 'Community interpreting',
    scenGroupSight: 'Sight translation-oriented',
    scenGroupInstitutions: 'International institutions',
    scenPanel: 'Panel discussion',
    scenLiveTV: 'Live TV broadcast',
    scenLegalCourt: 'Legal / Court (tribunal, police, deposition)',
    scenMedical: 'Medical / Healthcare (clinic, hospital, therapy)',
    scenPublicService: 'Public service (legal or medical consultation)',
    scenSecurityCouncil: 'UN Security Council',
    scenEcosoc: 'ECOSOC (Economic and Social Council)',
    scenWho: 'WHO (World Health Organization)',
    scenIlo: 'ILO (International Labour Organization)',
    scenUnesco: 'UNESCO',
    scenOif: 'OIF (Francophonie)',
    scenAuf: 'AUF (Francophone universities)',
    termDensity: 'Terminology density',
    optTermLow: 'Low — everyday vocabulary',
    optTermMedium: 'Medium',
    optTermHigh: 'High — dense specialised terms',
    longTextAsSource: 'Long text detected — it will be used as the source document and the speech will be grounded in it.',
    mcqPickAnswer: 'Select your answer, then check it.',
    mcqCheck: 'Check answer',
    mcqShowAnswer: 'Show answer',
    mcqCorrectMsg: '✓ Correct!',
    mcqWrongMsg: '✗ Not quite. Correct answer:',
    mcqShowAnswerLabel: 'ℹ️ Correct answer:',
    mcqYourScore: 'Your score',
    glossaryEdit: '✏️ Edit glossary',
    glossaryAddTerm: 'Add term',
    glossaryEditDone: '✓ Done editing',
    glossaryEditHint: 'Equivalents are looked up in the official bases (UNBIS Thesaurus, IATE, FranceTerme) — a ✓ marks a term confirmed there. Review and correct them BEFORE recording: the evaluation checks your terminology against this glossary.',
    termVerifiedTitle: 'Confirmed in',
    termEditedTitle: 'your correction',
    glossaryMemoryNote: 'corrections remembered — they are re-applied automatically to future glossaries.',
    glossaryMemoryClear: 'Forget saved corrections',
    webPageTab: 'Web page',
    webPageUrlPlaceholder: 'https://… paste the address of an article or report',
    webPageFetch: 'Fetch page',
    webPageFetching: 'Fetching page…',
    webPageAttached: 'Source attached',
    webPageDone: 'Done',
    fileTypeError: 'Unsupported file type. Upload a PDF, Word (.docx), or plain-text (.txt) file.',
    webPageHint: 'The readable text of the page will be extracted and used as the source for the speech.',
    scrollerTitle: 'Sight translation — scrolling text',
    scrollerPlay: '▶ Start scrolling',
    scrollerPause: '⏸ Pause',
    scrollerReset: '↺ Reset',
    scrollerSpeed: 'Speed (words/min)',
    scrollerFontSize: 'Text size',
    scrollerColumns: 'Columns',
    scrollerSpacing: 'Line spacing',
    scrollerHint: 'Translate aloud while the text scrolls at your chosen speed — like a real sight translation exercise.',
    notesTitle: '📝 Notes (consecutive)',
    notesTextTab: 'Text notes',
    notesSketchTab: 'Sketch / mind map',
    notesClear: 'Clear',
    notesShowSource: 'Show source',
    notesHideSource: 'Hide source',
    liveTranscriptOn: 'Live transcript',
    liveTranscriptOff: 'Hide live transcript',
    liveTranscriptTitle: 'Live transcription',
    liveTranscriptWaiting: 'Press play on the source audio — the transcription will appear here as it plays.',
    liveTranscriptHint: 'Split screen: the source speech is transcribed on one side as it plays; take your notes on the other. Play, pause, and rewind the source audio above to control it.',
    sourceText: 'Source',
    notesPlaceholder: 'Take your notes here while listening (symbols, arrows, keywords)…',
    notesHint: 'Your notes stay on this page only — they are not saved or sent anywhere. The sketch pad supports a stylus (pressure-sensitive).',
    simulTitle: '🎧 Simultaneous mode',
    simulStart: '▶ Play source + record me',
    simulStop: '⏹ Stop both',
    simulHint: 'Wear headphones so the source audio is not captured by your microphone. The source speech plays while your interpretation is recorded — real simultaneous practice.',
    simulNeedsAudio: 'Generate the source audio in "Audio & materials" first to enable simultaneous mode.',
    prepTimeLabel: 'preparation time before speaking — not counted against fluency',
    fluencyExplain: 'This score is measured from your recording: speaking rate, pauses during delivery, and how clearly each word was recognized. Time you take to prepare before speaking is NOT counted. A word the recognizer heard unclearly is treated as an audio-clarity signal, not automatically as a mistake.',
    aboutEvalTitle: 'ℹ️ How this evaluation works & where your data goes',
    aboutEvalCriteria: 'What is analyzed: (1) your recording is transcribed by a speech recognizer (Whisper); (2) measurable events are detected directly from the audio — pauses, repetitions, hesitations, speaking rate; (3) an AI evaluator compares your interpretation to the source speech meaning-by-meaning: accuracy, coverage, terminology, numbers and dates, grammar; (4) scores follow a strict professor rubric (most student performances score 4–7/10).',
    aboutEvalReliability: 'Reliability: the transcript is an estimate, not ground truth — especially for Arabic. Unclear audio regions are treated as clarity issues, not automatically as translation errors. Use this report as a training aid, not a final grade.',
    aboutEvalStorage: 'Storage: Your audio recording is processed and then deleted from the server — it is not kept. Only the evaluation summary (scores, error counts, recommendations) is saved to your account history. Generated speeches and transcripts are not stored permanently, but you can copy or download them.',
    creditsLine: 'ETIB · ESIB · CINIA · USJ Beirut',
    modeChoose: 'Practice mode',
    modeSimulDesc: 'The source speech plays while you interpret and record at the same time.',
    modeConsecDesc: 'Listen to the source and take notes, then record your interpretation.',
    modeSightDesc: 'Read the scrolling text and translate aloud while recording yourself.',
    heroTrainIn: 'Train in:',
    heroTitlePre: 'Master interpretation with ',
    heroTitleEm: 'AI-generated',
    heroTitlePost: ' speeches',
    heroParagraph: 'An adaptive training platform that generates speeches simulating authentic communication situations, builds multilingual glossaries, and evaluates your interpretation performance across Arabic, French, and English.',
    guestBtn: 'Continue as Guest',
    orSep: '— or —',
    srcPanelTitle: 'Add Source',
    srcPanelSubtitle: 'Find a UN document, a web page, or upload your own file to ground the speech in real content.',
    srcTabUN: 'UN Library',
    srcTabUpload: 'Upload file',
    srcUNHint: 'Search the UN Digital Library for your topic, download the PDF, then upload it below.',
    srcUNSearchBtn: 'Search UN Library ↗',
    srcUNUploadAfter: 'Once you have downloaded the PDF, upload it here:',
    srcDropPdf: 'Drop the PDF here or',
    srcDropFile: 'Drop a file here or',
    srcBrowse: 'browse',
    srcFileTypes: 'PDF, Word (.docx), or plain text',
    audioFluencyTitle: 'Audio fluency from the recording',
    coverageScoreLabel: 'Coverage score',
    statSpeechRate: 'Speech rate',
    statLongPauses: 'Long pauses',
    statSilenceRatio: 'Silence ratio',
    statWordConfidence: 'Word clarity',
    verdictGood: 'Good',
    verdictAcceptable: 'Acceptable',
    verdictNeedsWork: 'Needs work',
    nextHintPre: '→ Go to',
    nextHintPost: 'to listen, review questions, and download the glossary.',
    transcriptReady: 'Transcript ready',
    chatGreeting: "Hi! I'm your ETIB assistant. Ask me anything about interpretation techniques, the platform, or terminology.",
    chatPlaceholder: 'Ask anything…',
    chatSend: 'Send',
    chatThinking: 'Thinking…',
    statAvgPerSession: 'avg / session',
    statTotal: 'total',
    statSessions: 'Sessions',
    outsideRange: '(outside requested range)',
    progressGuestNote: 'Progress tracking works with an account. Sign in (or create a free account) so your evaluations are saved and analyzed across sessions.',
    pnTitle: 'Proper nouns (people, organizations, places)',
    pnCorrect: 'Correct',
    pnDistorted: 'Distorted — likely mispronounced',
    pnMissing: 'Omitted',
    llmFailedWarning: 'AI language analysis could not complete — showing automatic detections only. Scores may be incomplete.',
    llmQuotaError: 'The AI service hit its per-minute limit — this can happen with long speeches or several requests in a row. The platform already retried automatically; just wait about a minute and try again. The shared limit resets on its own — nothing to configure.',
    keyInvalid: 'The AI service rejected the request (the platform\'s server key is invalid or expired). This is not something you can fix — please tell your instructor or the platform administrator.',
    sameLangError: 'The source and interpretation languages must be different — interpretation is always between two languages.',
    simulSourceEnded: 'The source speech has ended — finish your interpretation, then press stop. Recording stops automatically after one minute.',
    simulWithText: 'Show source text on screen (SIMUL with text)',
    consecFlowHint: 'Step 1: listen to the speech and take notes. Step 2: record your interpretation from your notes below.',
    simulUseButton: 'Use "Play source + record me" above to start.',
    sightUseButton: 'Use the record button next to the scroller above to start.',
    downloadRecording: 'Download recording',
    downloadAudio: 'Download audio',
    llmDownError: 'The AI evaluation service did not respond — try again in a moment.',
    addSource: 'Add source',
    uploadSourceMedia: 'Upload a speech to interpret',
    sourceMediaEmpty: 'Could not transcribe the uploaded media — try a clearer recording.',
    micDenied: 'Microphone access denied — please allow permission and try again.',
    simultaneousFailed: 'Could not start simultaneous mode — check microphone permission.',
    topErrorSaid: 'said',
    topErrorExpected: 'Expected',
    topErrorMissing: 'Missing',
    uploadAudioBtn: 'Upload audio file',
    correctTashkeel: 'Correct tashkeel',
    iraabTitle: 'Arabic case endings (i\'rab)',
    iraabAcousticNote: 'Detected acoustically from your recording — the transcript is kept verbatim and wrong endings are not corrected.',
    iraabDetected: 'Heard',
    iraabCorrect: 'Correct',
    iraabIncorrect: 'Incorrect',
    iraabUncertain: 'Uncertain',
    iraabAccuracy: 'Ending accuracy',
    iraabDiacritized: 'Vocalized hypothesis',
  },
  ar: {
    uiLanguage: 'لغة الواجهة',
    loginTitle: 'تسجيل الدخول إلى منصة ETIB',
    loginSubtitle: 'ادخل إلى مساحة التدريب على الترجمة الفورية.',
    signupTitle: 'إنشاء حساب في منصة ETIB',
    signupSubtitle: 'أنشئ حساباً للبدء بالتدريب على الترجمة الفورية.',
    name: 'الاسم الكامل',
    email: 'البريد الإلكتروني',
    password: 'كلمة المرور',
    showPassword: 'إظهار',
    hidePassword: 'إخفاء',
    role: 'الدور',
    loginSubmit: 'تسجيل الدخول',
    loginLoading: 'جارٍ تسجيل الدخول...',
    signupSubmit: 'إنشاء الحساب',
    signupLoading: 'جارٍ إنشاء الحساب...',
    loginNote: 'استخدم بيانات حسابك، أو أنشئ حساباً جديداً.',
    switchToSignup: 'إنشاء حساب جديد',
    switchToLogin: 'لديك حساب؟ سجّل الدخول',
    student: 'طالب',
    guestName: 'ضيف',
    instructor: 'مدرب',
    coordinator: 'منسق',
    signOut: 'تسجيل الخروج',
    workspaceKicker: 'وحدات التدريب',
    workspaceTitle: 'توليد الخطابات، إعداد المواد، تسجيل الأداء، ومراجعة التغذية الراجعة.',
    navA: 'توليد الخطاب',
    navB: 'الصوت والمواد',
    navC: 'التسجيل والتفريغ',
    navD: 'التقييم',
    navE: 'التقدم',
    progressTitle: 'تقدمي',
    progressSubtitle: 'سجل الدرجات والتوصيات التكيفية',
    progressNoSessions: 'لا جلسات بعد — أكمل تقييماً كاملاً لبدء تتبع تقدمك.',
    progressSessions: 'سجل الجلسات',
    progressAdaptive: 'التوصيات التكيفية',
    progressAdaptiveSubtitle: 'بناءً على جلساتك الأخيرة، توصي المنصة بما يلي:',
    progressApply: 'تطبيق هذه الإعدادات',
    progressOverall: 'إجمالي',
    progressFluency: 'الطلاقة',
    progressCoverage: 'التغطية',
    progressErrors: 'الأخطاء',
    progressLang: 'اللغة',
    progressDate: 'التاريخ',
    progressScore: 'الدرجة',
    progressLoading: 'جارٍ تحميل السجل…',
    progressAvgOver: 'متوسط الإجمالي',
    progressAvgFluency: 'متوسط الطلاقة',
    progressAvgNumbers: 'متوسط أخطاء الأرقام',
    progressTrend: 'منحنى الدرجات (آخر 10 جلسات)',
    progressLatest: 'آخر جلسة',
    progressProblems: 'نقاط التركيز — ما يجب العمل عليه',
    progressImproving: 'تحسّن',
    progressDeclining: 'تراجع',
    progressStable: 'مستقر',
    progressStrengths: 'نقاط القوة',
    progressRecs: 'التوصيات',
    severityHigh: 'مرتفعة', severityMedium: 'متوسطة', severityLow: 'منخفضة',
    problemLabel_translation: 'دقة الترجمة',
    problemLabel_numbers: 'الأرقام والإحصاءات',
    problemLabel_missing: 'المحتوى الفائت',
    problemLabel_silences: 'التوقفات الطويلة',
    problemLabel_repetitions: 'التكرارات والتصحيحات',
    problemLabel_fluency: 'طلاقة الإلقاء',
    problemLabel_coverage: 'تغطية المحتوى',
    problemTip_translation: 'راجع قوائم المصطلحات ومارس الاستماع الفعّال للمفاهيم الأساسية.',
    problemTip_numbers: 'تدرّب على محاذاة الخطابات ذات الكثافة العالية من الأرقام.',
    problemTip_missing: 'طوّر مهاراتك في تدوين الملاحظات وابدأ بخطابات أقصر.',
    problemTip_silences: 'مارس تمارين التأخير الزمني للحفاظ على انسيابية الإخراج.',
    problemTip_repetitions: 'اعمل على إعادة الصياغة — التزم بترجمتك بدلاً من التراجع عنها.',
    problemTip_fluency: 'استخدم المحاذاة الصوتية مع متحدثين أصليين وسجّل نفسك يومياً.',
    problemTip_coverage: 'ركّز على نقل الأفكار الرئيسية قبل الاهتمام بالأسلوب.',
    problemDetail_translation: 'أخطاء في الترجمة في {count}/{total} جلسات',
    problemDetail_numbers: 'أخطاء في الأرقام في {count}/{total} جلسات',
    problemDetail_missing: 'أفكار مفقودة في {count}/{total} جلسات',
    problemDetail_silences: 'توقفات مفرطة في {count}/{total} جلسات',
    problemDetail_repetitions: 'تكرارات مرتفعة في {count}/{total} جلسات',
    problemDetail_fluency: 'درجة طلاقة منخفضة في {count}/{total} جلسات',
    problemDetail_coverage: 'تغطية منخفضة في {count}/{total} جلسات',
    progressTopErrors: 'أخطاء محددة',
    progressNoData: 'بيانات غير كافية بعد',
    moduleATitle: 'توليد خطاب تدريبي',
    groundedSourceLabel: 'مستند إلى مصدر حقيقي:',
    ungroundedNote: 'لم يتم العثور على وثيقة أممية مطابقة — تم إنشاء هذا الخطاب دون الاستناد إلى مصدر.',
    language: 'لغة الخطاب',
    targetLanguage: 'لغة الترجمة الهدف',
    topic: 'الموضوع',
    domain: 'المجال',
    wordCount: 'طول الخطاب',
    wordRangeShort: 'قصير',
    wordRangeMedium: 'متوسط',
    wordRangeLong: 'طويل',
    difficulty: 'مستوى الصعوبة',
    structure: 'بنية الخطاب',
    mode: 'نوع الترجمة الفورية',
    numbers: 'كثافة الأرقام',
    hesitationsSim: 'محاكاة التردد',
    pressureMode: 'وضع الضغط',
    speedPressure: 'ضغط السرعة',
    topicShifts: 'تحولات الموضوع',
    contextNoise: 'ضوضاء السياق',
    cognitiveLoad: 'العبء المعرفي',
    summary: 'ملخص',
    mcq: 'أسئلة متعددة الخيارات',
    glossary: 'المسرد',
    documentGrounding: 'التوليد من وثيقة',
    documentFile: 'الوثيقة المصدر',
    sharedSettings: 'إعدادات التوليد',
    sharedSettingsHint: 'تُطبَّق هذه الإعدادات على التوليد بالموضوع أو بالوثيقة.',
    generationMethods: 'اختر مصدر التوليد',
    topicOnlyGeneration: 'توليد بالموضوع فقط',
    topicOnlyHint: 'توليد الخطاب من الموضوع والإعدادات المحددة.',
    documentGenerationHint: 'ارفع وثيقة لتوليد خطاب مستند إلى محتواها.',
    generateFromDocument: '▷ توليد من وثيقة',
    generationEmpty: 'لم يُولَّد أي نص للخطاب (قد يكون الخادم في طور الاستيقاظ أو جرى تقييد الطلب). لم يُحفظ شيء — يُرجى المحاولة مرة أخرى.',
    retrieveContext: 'معاينة السياق المستخرج',
    retrievedContext: 'السياق المستخرج',
    submit: 'توليد الخطاب',
    generating: 'جارٍ التوليد، يرجى الانتظار...',
    wordsUnit: 'كلمة',
    moduleBTitle: 'الصوت والمواد التعليمية',
    moduleBBody: 'إنتاج الصوت، المصطلحات الأساسية، المسرد، والأسئلة متعددة الخيارات.',
    moduleCTitle: 'التسجيل والتفريغ',
    moduleCBody: 'تسجيل داخل المتصفح، تفريغ آلي، ونص زمني.',
    moduleDTitle: 'تقييم الأداء',
    moduleDBody: 'رصد التردد، الحذف، أخطاء الأرقام، والمشاكل المصطلحية.',
    comingSoon: 'قريباً',
    errorPrefix: 'خطأ',
    noSpeechYet: 'ولّد خطاباً في الوحدة أ أولاً، ثم عد إلى هنا.',
    materialsMissing: 'أسئلة الفهم والمسرد لهذا الخطاب غير موجودة (قد تختفي في الخطابات الطويلة جداً). ولّدها الآن:',
    materialsGenerating: 'جارٍ التوليد…',
    voiceAccent: 'نبرة الصوت',
    speechRate: 'سرعة الإلقاء',
    sessionRestored: 'استُعيدت جلستك السابقة (الخطاب والمواد وتعديلات المسرد). يمكنك المتابعة من حيث توقفت.',
    sessionKeep: 'متابعة',
    sessionDiscard: 'البدء من جديد',
    measuredRate: 'السرعة المقاسة',
    wordsPerMinute: 'كلمة/دقيقة',
    rateTooFastHint: 'أعلى من النطاق 100–120 كلمة/دقيقة الموصى به للترجمة الفورية — خفّض سرعة الإلقاء وأعد التوليد',
    mediaUnreadable: 'تعذّرت قراءة هذا الملف. الصيغ المدعومة: MP3 وWAV وM4A وOGG وWebM وMP4. إذا كان الملف يعمل على جهازك فحوّله إلى MP3 أو WAV وأعد المحاولة.',
    generateAudio: 'توليد الصوت',
    generatingAudio: 'جارٍ توليد الصوت...',
    materialsTitle: 'المواد التعليمية',
    generateMaterials: 'توليد المواد',
    generatingMaterials: 'جارٍ توليد المواد...',
    materialsFailed: 'تم تفريغ الخطاب، لكن تعذّر توليد المواد (الملخص، الأسئلة، المسرد) — قد يكون الخادم قيد التشغيل (٢-٣ دقائق). اضغط «رفع خطاب» مجدداً لإعادة المحاولة.',
    keyTerms: 'المصطلحات الأساسية',
    summaryTitle: 'الملخص الموضوعي',
    mcqTitle: 'أسئلة الفهم (اختيار متعدد)',
    mcqQuestionHeader: 'السؤال',
    mcqOptionsHeader: 'الخيارات',
    mcqAnswerHeader: 'الإجابة',
    glossaryTitle: 'المسرد الثلاثي (عربي – فرنسي – إنجليزي)',
    glossaryDuringRecording: 'المسرد (أبقِ المصطلحات ظاهرة أثناء الترجمة)',
    glossaryTermHeader: 'المصطلح',
    glossaryArabicHeader: 'العربية',
    glossaryFrenchHeader: 'الفرنسية',
    glossaryEnglishHeader: 'الإنجليزية',
    glossaryDefinitionHeader: 'التعريف',
    downloadGlossary: 'تنزيل المسرد (DOCX)',
    uploadGlossary: 'استيراد مسرد',
    glossaryUploading: 'جارٍ الاستيراد…',
    glossaryUploadEmpty: 'تعذّرت قراءة أي مصطلحات من الملف. استخدم أعمدة مثل: المصطلح / العربية / الفرنسية / الإنجليزية (بصيغة CSV أو TSV أو TXT أو DOCX أو JSON).',
    uploadAudio: 'ارفع تسجيل ترجمتك (MP3, WAV, M4A)',
    transcribeBtn: 'فرّغ الصوت',
    transcribing: 'جارٍ التفريغ — قد يستغرق دقيقة أو دقيقتين...',
    transcriptLabel: 'النص المفرَّغ',
    lowConfidence: 'كلمات لم يتبيّنها نظام التعرّف بوضوح (راجِعها)',
    arabicTip: 'للحصول على أفضل النتائج: تحدث بالعربية الفصحى في غرفة هادئة.',
    noTranscriptYet: 'سجّل ترجمتك أولاً في "التسجيل والتفريغ"، ثم عد إلى هنا.',
    runEvaluation: 'تشغيل التقييم الكامل',
    runningEvaluation: 'جارٍ تحليل أدائك…',
    overallScore: 'الدرجة الإجمالية',
    evalSummary: 'التقييم العام',
    strengths: 'نقاط القوة',
    recommendations: 'التوصيات',
    algorithmicTitle: 'الرصد الآلي',
    longSilences: 'فترات صمت طويلة',
    repetitions: 'تكرارات',
    hesitations: 'ترددات',
    filledPause: 'تردد صوتي (اممم…)',
    cutWord: 'كلمة مقطوعة',
    numberErrors: 'أخطاء في الأرقام',
    llmTitle: 'التحليل اللغوي بالذكاء الاصطناعي',
    languageErrors: 'أخطاء لغوية',
    autoCorrections: 'تصحيحات ذاتية',
    falseStarts: 'بدايات مبتورة',
    lapsusLinguae: 'زلّات لسان',
    terminologyProblems: 'مشكلات في المصطلحات',
    informationLoss: 'فقدان معلومات',
    noIssues: 'لا توجد مشكلات',
    correction: 'التصحيح',
    source: 'المصدر',
    used: 'ما قيل',
    correct: 'المصطلح الصحيح',
    coverageTitle: 'تغطية المحتوى',
    coverageHint: 'نسبة ما نُقل من مضمون الخطاب المصدر',
    translationErrors: 'أخطاء الترجمة',
    meaningCheckTitle: 'التحقق من المعنى',
    meaningCheckNote: 'تغييرات محتملة في المعنى — تُعرض لمساعدتك على التعلّم ولا تؤثر في علامتك. يمكن التعبير عن الفكرة نفسها بكلمات مختلفة، لذا لا يُذكر سوى التغييرات الفعلية في المعنى.',
    sourceSaid: 'ما ورد في المصدر',
    studentSaid: 'ما قلتَه',
    correctTranslation: 'الترجمة الصحيحة',
    missingContent: 'محتوى مفقود',
    scoreScopeNote: 'تعكس هذه الدرجة الالتزام بالمصطلحات، وصحّة اللغة الهدف، والأرقام والأسماء، والأداء (التوقّفات، إدارة الفارق الزمني، السلاسة). أمّا المعنى/التغطية المعروضان أدناه فهما للاطّلاع فقط ولا يؤثّران في الدرجة في هذه المرحلة.',
    aiDisclaimerTitle: '⚠️ تنبيه بشأن استخدام الذكاء الاصطناعي في هذه المنصّة',
    aiDisclaimerShort: 'قد يخطئ الذكاء الاصطناعي — تحقّق من النص والصوت والتفريغ والدرجات قبل الاعتماد عليها.',
    aiDisclaimerMore: 'التفاصيل',
    aiDisclaimerLess: 'إغلاق',
    aiDisclaimerGen: 'النصوص المولَّدة: تُولَّد النصوص في هذه المنصّة بواسطة نموذج لغوي كبير، وبالتالي قد تتضمّن أرقاماً مختلَقة، وعناوين تقارير أو قرارات غير موجودة، وأخطاءً في البيانات، وانحيازات ثقافية أو سياسية. لا يمكن التعويل عليها كمصدر للمعلومات؛ إنّما هي موادّ معدّة للتدريب. كما يجب التنبّه إلى الأعداد وحركات الإعراب والتواريخ، لأنّها الأكثر عرضةً للخطأ عند توليد المحتوى باللغة العربية.',
    aiDisclaimerTts: 'الصوت الاصطناعي: قد يخطئ الصوت المولَّد في نطق الأسماء والاختصارات والتواريخ والأرقام، ولا سيّما باللغة العربية. وقد لا يتطابق ما تسمعه تماماً مع النص المكتوب. وإذا بدا نطق أحد المقاطع غير سليم، يُرجى تنقيح النص قبل توليد التسجيل الصوتي.',
    aiDisclaimerStt: 'التفريغ الصوتي: يُحوَّل التسجيل الصوتي تلقائياً إلى نص مكتوب. ويُعدّ هذا النص تفريغاً تقديرياً لما تمّ نطقه، ولا ينبغي اعتباره سجلاً حرفياً أو مؤكَّداً للكلام المسجَّل. كما أنّ استخدام اللغة العربية، أو التنقّل بين لغات متعدّدة، أو وجود ضوضاء وأصوات في الخلفية، كلّها عوامل قد تؤثّر بصورة ملحوظة في دقة التفريغ. لذلك، يُرجى مراجعة نص التفريغ وتصحيحه قبل إجراء عملية التقييم.',
    aiDisclaimerEval: 'التقييم: التقرير أداة تدريب آلية، وليس علامة أو بديلاً عن أستاذك. وهو لا يقيّم أداءك إلا حسب المعايير المذكورة أدناه، وما عداها إرشادي.',
    aiDisclaimerAdvice: 'يُنصح بالتحقّق من المصطلحات في قواعد البيانات المرجعية المعتمدة، ومراجعة الأرقام وأسماء العلم ومقارنتها بالنص المصدر، وتصحيح النص المفرَّغ، وإبلاغ الأستاذ بأي معلومات تبدو خاطئة.',
    aiDisclaimerAck: '\u0641\u0647\u0645\u062a',
    ownTextBtn: '\u270d\ufe0f \u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0646\u0635\u0651\u064a \u0627\u0644\u062e\u0627\u0635 (\u0628\u0644\u0627 \u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646\u0627\u0639\u064a)',
    ownTextTitle: '\u270d\ufe0f \u0646\u0635\u0651\u064a \u0627\u0644\u062e\u0627\u0635 \u2014 \u0635\u0648\u062a \u0641\u0642\u0637 \u0628\u0644\u0627 \u062a\u0648\u0644\u064a\u062f',
    ownTextHint: '\u0627\u0644\u0635\u0642 \u0646\u0635\u0651\u0627\u064b \u0644\u062f\u064a\u0643 \u0645\u0633\u0628\u0642\u0627\u064b. \u0633\u064a\u064f\u0633\u062a\u062e\u062f\u0645 \u0643\u0645\u0627 \u0647\u0648 \u062a\u0645\u0627\u0645\u0627\u064b: \u0628\u0644\u0627 \u0625\u0639\u0627\u062f\u0629 \u0635\u064a\u0627\u063a\u0629 \u0648\u0644\u0627 \u062a\u0648\u0644\u064a\u062f. \u062b\u0645\u0651 \u0627\u0646\u062a\u0642\u0644 \u0625\u0644\u0649 \u00ab\u0627\u0644\u0635\u0648\u062a \u0648\u0627\u0644\u0645\u0648\u0627\u062f\u00bb \u0644\u062a\u062d\u0648\u064a\u0644\u0647 \u0625\u0644\u0649 \u0635\u0648\u062a \u062b\u0645\u0651 \u062a\u0631\u062c\u0645\u0647 \u0643\u0627\u0644\u0645\u0639\u062a\u0627\u062f.',
    ownTextPlaceholder: '\u0627\u0644\u0635\u0642 \u0646\u0635\u0651\u0643 \u0647\u0646\u0627\u2026',
    ownTextUse: '\u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0647\u0630\u0627 \u0627\u0644\u0646\u0635 \u0643\u0645\u0627 \u0647\u0648',
    ownTextEmpty: '\u0627\u0644\u0635\u0642 \u0646\u0635\u0651\u0627\u064b \u0623\u0648\u0651\u0644\u0627\u064b.',
    ownTextBadge: '\u0646\u0635\u0651\u0643 \u0627\u0644\u062e\u0627\u0635 \u2014 \u063a\u064a\u0631 \u0645\u0648\u0644\u0651\u064e\u062f \u0628\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a',
    ownTextMaterialsHint: '\u0623\u064f\u0636\u064a\u0641 \u0647\u0630\u0627 \u0627\u0644\u0646\u0635 \u0643\u0645\u0627 \u0647\u0648\u060c \u0641\u0644\u0627 \u064a\u062a\u0636\u0645\u0651\u0646 \u0645\u0644\u062e\u0651\u0635\u0627\u064b \u0648\u0644\u0627 \u0623\u0633\u0626\u0644\u0629 \u0627\u062e\u062a\u064a\u0627\u0631 \u0645\u0646 \u0645\u062a\u0639\u062f\u0651\u062f \u0648\u0644\u0627 \u0645\u0633\u0631\u062f\u0627\u064b. \u0648\u0644\u0651\u062f\u0647\u0627 \u0628\u0627\u0644\u0632\u0631 \u0625\u0646 \u0623\u0631\u062f\u062a \u2014 \u0648\u0644\u0646 \u064a\u062a\u063a\u064a\u0651\u0631 \u0627\u0644\u0646\u0635 \u0646\u0641\u0633\u0647.',
    scriptTitle: '\u0646\u0635 \u0627\u0644\u062e\u0637\u0627\u0628',
    editScriptBtn: '\u270e \u062a\u0639\u062f\u064a\u0644 \u0627\u0644\u0646\u0635',
    editScriptDone: '\u2713 \u062a\u0645\u0651',
    editScriptHint: '\u0635\u062d\u0651\u062d \u0627\u0644\u0646\u0635 \u0642\u0628\u0644 \u062a\u0648\u0644\u064a\u062f \u0627\u0644\u0635\u0648\u062a \u2014 \u0627\u0644\u0623\u062e\u0637\u0627\u0621 \u0648\u0627\u0644\u0623\u0631\u0642\u0627\u0645 \u063a\u064a\u0631 \u0627\u0644\u0645\u0648\u0641\u0651\u0642\u0629 \u0648\u0627\u0644\u0623\u0633\u0645\u0627\u0621 \u0627\u0644\u062e\u0627\u0637\u0626\u0629. \u064a\u0639\u062a\u0645\u062f \u0627\u0644\u0635\u0648\u062a \u0648\u0627\u0644\u062a\u0631\u062c\u0645\u0629 \u0648\u0627\u0644\u062a\u0642\u064a\u064a\u0645 \u062c\u0645\u064a\u0639\u0627\u064b \u0639\u0644\u0649 \u0647\u0630\u0647 \u0627\u0644\u0646\u0633\u062e\u0629 \u0627\u0644\u0645\u0635\u062d\u0651\u062d\u0629.',
    scriptRestore: '\u21ba \u0627\u0633\u062a\u0639\u0627\u062f\u0629 \u0627\u0644\u0646\u0635 \u0627\u0644\u0623\u0635\u0644\u064a',
    editTranscriptBtn: '\u270e \u062a\u0635\u062d\u064a\u062d \u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0641\u0631\u0651\u064e\u063a',
    editTranscriptDone: '\u2713 \u062d\u0641\u0638 \u0627\u0644\u062a\u0635\u062d\u064a\u062d',
    editTranscriptHint: '\u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0641\u0631\u0651\u064e\u063a \u0647\u0648 \u0645\u0627 \u0633\u0645\u0639\u0647 \u0646\u0638\u0627\u0645 \u0627\u0644\u062a\u0639\u0631\u0651\u0641 \u0639\u0644\u0649 \u0627\u0644\u0643\u0644\u0627\u0645. \u0635\u062d\u0651\u062d \u0623\u064a \u062e\u0637\u0623 \u0641\u064a \u0627\u0644\u062a\u0639\u0631\u0651\u0641 \u0642\u0628\u0644 \u0627\u0644\u062a\u0642\u064a\u064a\u0645 \u0643\u064a \u0644\u0627 \u064a\u064f\u062d\u0633\u064e\u0628 \u062e\u0637\u0623 \u0627\u0644\u0622\u0644\u0629 \u0639\u0644\u064a\u0643. \u062a\u0628\u0642\u0649 \u0627\u0644\u062a\u0648\u0642\u0651\u0641\u0627\u062a \u0648\u0627\u0644\u062a\u0631\u062f\u0651\u062f \u0648\u0633\u0631\u0639\u0629 \u0627\u0644\u0625\u0644\u0642\u0627\u0621 \u0645\u0642\u064a\u0633\u0629 \u0645\u0646 \u062a\u0633\u062c\u064a\u0644\u0643 \u0627\u0644\u0641\u0639\u0644\u064a.',
    transcriptEditedBadge: '\u0646\u0635 \u0645\u0641\u0631\u0651\u064e\u063a \u0635\u062d\u0651\u062d\u062a\u064e\u0647 \u0628\u0646\u0641\u0633\u0643',
    micWarmingUp: '\u062c\u0627\u0631\u064d \u062a\u0647\u064a\u0626\u0629 \u0627\u0644\u0645\u064a\u0643\u0631\u0648\u0641\u0648\u0646\u2026',
    micReadyHint: '\u0627\u0646\u062a\u0638\u0631 \u0638\u0647\u0648\u0631 \u00ab\u062c\u0627\u0631\u064d \u0627\u0644\u062a\u0633\u062c\u064a\u0644\u00bb \u0642\u0628\u0644 \u0623\u0646 \u062a\u0628\u062f\u0623 \u0627\u0644\u0643\u0644\u0627\u0645.',
    mediaAsrWarning: '\u0647\u0630\u0627 \u0627\u0644\u0646\u0635 \u0646\u0627\u062a\u062c \u0639\u0646 \u062a\u0641\u0631\u064a\u063a \u0622\u0644\u064a \u0644\u0645\u0644\u0641\u0643. \u062a\u062a\u0648\u0642\u0651\u0641 \u062f\u0642\u0651\u062a\u0647 \u0639\u0644\u0649 \u062c\u0648\u062f\u0629 \u0627\u0644\u0635\u0648\u062a (\u0648\u0647\u064a \u0623\u062f\u0646\u0649 \u0641\u064a \u0627\u0644\u0639\u0631\u0628\u064a\u0629): \u0623\u0639\u062f \u0642\u0631\u0627\u0621\u062a\u0647 \u0648\u0635\u062d\u0651\u062d\u0647 \u0641\u064a \u00ab\u0627\u0644\u0635\u0648\u062a \u0648\u0627\u0644\u0645\u0648\u0627\u062f\u00bb \u0642\u0628\u0644 \u0627\u0639\u062a\u0645\u0627\u062f\u0647 \u0645\u0635\u062f\u0631\u0627\u064b.',
    fluencyScore: 'درجة الطلاقة',
    pauseAnalysisTitle: 'التوقّفات والانسياب',
    repetitionAnalysisTitle: 'تحليل التكرارات',
    numberAccuracyTitle: 'الأرقام والتواريخ',
    pronunciationAssessmentTitle: 'ملاحظات على النطق',
    numCorrect: 'صحيح',
    numIncorrect: 'غير صحيح',
    expectedLabel: 'الصواب',
    studentSaidShort: 'ما قلتَه',
    pronunciationTitle: 'تقييم النطق',
    pronunciationMismatches: 'فروق الوضوح على مستوى الكلمة',
    runPronunciation: 'إجراء فحص النطق',
    runningPronunciation: 'جارٍ محاذاة الصوت بالنص المصدر — قد يستغرق ذلك دقيقة…',
    pronunciationScore: 'مدى تيقُّن النظام',
    wordView: 'عرض كلمة بكلمة',
    legend: 'مفتاح الألوان',
    legendGood: '80% فأكثر — واضحة',
    legendWarn: '65–80%',
    legendPoor: 'أقل من 65% — خطأ محتمل',
    uncertainWords: 'كلمات أساسية بدت غير واضحة',
    expectedForm: 'الشكل الصحيح',
    likelyError: 'الخطأ المحتمل',
    grammaticalRole: 'الدور النحوي',
    noUncertain: 'لم ترد كلمات أساسية غير واضحة',
    whisperxBadge: 'محاذاة WhisperX',
    fallbackBadge: 'نتائج Whisper',
    needsArabic: 'فحص النطق متاح للترجمات العربية فقط.',
    needsSource: 'ولّد خطاباً في الوحدة أ أولاً لتوفير النص المرجعي.',
    sourceAudio: 'الصوت المرجعي للخطاب',
    playbackSpeed: 'سرعة التشغيل',
    yourInterpretation: 'ترجمتك الفورية',
    recordBtn: '🎙 بدء التسجيل',
    stopBtn: '⏹ إيقاف التسجيل',
    recordingLive: 'جارٍ التسجيل',
    autoTranscribe: 'تفريغ تلقائي بعد التسجيل',
    orUpload: 'أو ارفع ملف صوتي',
    deleteAudio: 'حذف التسجيل',
    redoTranscript: 'إعادة التفريغ',
    copyText: 'نسخ',
    bubbleGood: 'تفريغ جيد',
    bubblePoor: 'تفريغ رديء',
    deleteTranscript: 'حذف النص المفرَّغ',
    closeLabel: 'إغلاق',
    copied: 'تم النسخ!',
    topicPlaceholder: 'أدخل موضوعاً أو الصق نصاً لتوليد خطاب...',
    namePlaceholder: 'اسمك الكامل',
    librarySearchPlaceholder: 'مثال: تغير المناخ، حوكمة الذكاء الاصطناعي…',
    removeSource: 'إزالة المصدر',
    addSourceFirst: 'أضف مستنداً مصدرياً واحداً على الأقل أولاً.',
    moreSettings: 'المزيد ⚙',
    lessSettings: 'أقل ⚙',
    optConsecutive: 'تتابعي',
    optSimultaneous: 'فوري',
    optSight: 'ترجمة بصرية',
    optWellOrganized: 'منظَّم جيداً',
    optDisorganized: 'غير منظَّم',
    optLowNumbers: 'أرقام قليلة',
    optHighNumbers: 'أرقام كثيرة',
    optNormalSpeed: 'سرعة عادية',
    optFast: 'سريع',
    optVeryFast: 'سريع جداً',
    optNoShifts: 'بدون تحولات',
    optShifts: 'تحولات موضوعية',
    domPolitics: 'السياسة',
    domDiplomacy: 'الدبلوماسية',
    domEconomics: 'الاقتصاد',
    domHealth: 'الصحة',
    domEducation: 'التعليم',
    domClimate: 'المناخ والبيئة',
    domHumanRights: 'حقوق الإنسان',
    domTechnology: 'التكنولوجيا والذكاء الاصطناعي',
    domMigration: 'الهجرة واللاجئون',
    domDisarmament: 'نزع السلاح',
    domWomen: 'المرأة والنوع الاجتماعي',
    domFood: 'الغذاء والجوع',
    domLegal: 'القانون',
    domMedical: 'الطب',
    diffBeginner: 'مبتدئ',
    diffIntermediate: 'متوسط',
    diffAdvanced: 'متقدم',
    diffHint: 'مبتدئ: مفردات بسيطة، جمل قصيرة، رقم أو رقمان، إيقاع بطيء. متوسط: مصطلحات معتدلة، عدة إحصاءات، جمل متنوعة. متقدم: مصطلحات كثيفة، أرقام وأسماء متكررة، تراكيب معقدة، إيقاع سريع.',
    wordRangeShortFull: 'قصير — 120–180 كلمة (≈ دقيقة واحدة)',
    wordRangeMediumFull: 'متوسط — 220–320 كلمة (≈ 1.5–2.5 دقيقة)',
    wordRangeLongFull: 'طويل — 400–550 كلمة (≈ 3–4 دقائق)',
    wordRangeExtendedFull: 'ممتد — 650–800 كلمة (≈ 4.5–6 دقائق)',
    wordRangeVeryLongFull: 'طويل جداً — 1100–1400 كلمة (≈ 8–11 دقيقة)',
    wordRangeExtraLongFull: 'طويل إضافي — 1500–2200 كلمة (≈ 12–18 دقيقة)',
    wordRangeMarathonFull: 'ماراثوني — 2900–3600 كلمة (≈ 24–30 دقيقة)',
    optSemiStructured: 'شبه منظَّم',
    optDisorganizedFull: 'غير منظَّم عمداً',
    scenarioLabel: 'أسلوب المتحدث / السياق',
    scenUNGA: 'الجمعية العامة للأمم المتحدة',
    scenEUParl: 'البرلمان الأوروبي',
    scenArabLeague: 'قمة الجامعة العربية',
    scenPress: 'مؤتمر صحفي',
    scenDiplomatic: 'اجتماع دبلوماسي',
    scenDebate: 'مناظرة سياسية',
    scenInterview: 'مقابلة',
    scenGroupGeneral: 'عام / مؤتمرات',
    scenGroupSimul: 'مخصص للترجمة الفورية',
    scenGroupConsec: 'الترجمة المجتمعية',
    scenGroupSight: 'مخصص للترجمة المنظورة',
    scenGroupInstitutions: 'المنظمات الدولية',
    scenPanel: 'حلقة نقاش',
    scenLiveTV: 'بث تلفزيوني مباشر',
    scenLegalCourt: 'قضائي / محكمة (تحقيق، شرطة، إفادة)',
    scenMedical: 'طبي / رعاية صحية (عيادة، مستشفى، علاج نفسي)',
    scenPublicService: 'خدمة عامة (استشارة قانونية أو طبية)',
    scenSecurityCouncil: 'مجلس الأمن الدولي',
    scenEcosoc: 'المجلس الاقتصادي والاجتماعي (ECOSOC)',
    scenWho: 'منظمة الصحة العالمية',
    scenIlo: 'منظمة العمل الدولية',
    scenUnesco: 'اليونسكو',
    scenOif: 'المنظمة الدولية للفرنكوفونية',
    scenAuf: 'الوكالة الجامعية للفرنكوفونية',
    termDensity: 'الكثافة المصطلحية',
    optTermLow: 'منخفضة — مفردات يومية',
    optTermMedium: 'متوسطة',
    optTermHigh: 'عالية — مصطلحات متخصصة كثيفة',
    longTextAsSource: 'تم رصد نص طويل — سيُستخدم كوثيقة مصدر وسيُبنى الخطاب على محتواه.',
    mcqPickAnswer: 'اختر إجابتك ثم تحقق منها.',
    mcqCheck: 'تحقق من الإجابة',
    mcqShowAnswer: 'أظهر الإجابة',
    mcqCorrectMsg: '✓ صحيح!',
    mcqWrongMsg: '✗ غير صحيح. الإجابة الصحيحة:',
    mcqShowAnswerLabel: 'ℹ️ الإجابة الصحيحة:',
    mcqYourScore: 'نتيجتك',
    glossaryEdit: '✏️ تعديل المسرد',
    glossaryAddTerm: 'إضافة مصطلح',
    glossaryEditDone: '✓ إنهاء التعديل',
    glossaryEditHint: 'تُستمدّ المقابلات من القواعد الرسمية (مكنز الأمم المتحدة UNBIS ، IATE ، FranceTerme) — وتُشير علامة ✓ إلى مصطلح مؤكَّد فيها. راجعها وصحّحها قبل التسجيل: يقيس التقييم مصطلحاتك على هذا المسرد.',
    termVerifiedTitle: 'مؤكَّد في',
    termEditedTitle: 'تصحيحك',
    glossaryMemoryNote: 'تصحيحات محفوظة — تُطبَّق تلقائياً على المسارد القادمة.',
    glossaryMemoryClear: 'حذف التصحيحات المحفوظة',
    webPageTab: 'صفحة ويب',
    webPageUrlPlaceholder: '…https:// الصق رابط مقال أو تقرير',
    webPageFetch: 'جلب الصفحة',
    webPageFetching: 'جارٍ جلب الصفحة…',
    webPageAttached: 'تم إرفاق المصدر',
    webPageDone: 'تم',
    fileTypeError: 'نوع الملف غير مدعوم. ارفع ملف PDF أو Word‏ (.docx) أو نصاً عادياً‏ (.txt).',
    webPageHint: 'سيُستخرج النص المقروء من الصفحة ويُستخدم كمصدر للخطاب.',
    scrollerTitle: 'الترجمة البصرية — نص متحرك',
    scrollerPlay: '▶ بدء التمرير',
    scrollerPause: '⏸ إيقاف مؤقت',
    scrollerReset: '↺ إعادة',
    scrollerSpeed: 'السرعة (كلمة/دقيقة)',
    scrollerFontSize: 'حجم الخط',
    scrollerColumns: 'الأعمدة',
    scrollerSpacing: 'تباعد الأسطر',
    scrollerHint: 'ترجم بصوت عالٍ أثناء تمرير النص بالسرعة التي تختارها — تماماً كتمرين ترجمة بصرية حقيقي.',
    notesTitle: '📝 الملاحظات (تتابعي)',
    notesTextTab: 'ملاحظات نصية',
    notesSketchTab: 'رسم / خريطة ذهنية',
    notesClear: 'مسح',
    notesShowSource: 'إظهار المصدر',
    notesHideSource: 'إخفاء المصدر',
    liveTranscriptOn: 'التفريغ المباشر',
    liveTranscriptOff: 'إخفاء التفريغ المباشر',
    liveTranscriptTitle: 'التفريغ المباشر',
    liveTranscriptWaiting: 'شغّل الصوت المصدر — سيظهر التفريغ هنا أثناء التشغيل.',
    liveTranscriptHint: 'شاشة مقسّمة: يُفرَّغ الخطاب المصدر على جهة أثناء تشغيله، ودوّن ملاحظاتك على الجهة الأخرى. تحكّم بذلك عبر تشغيل الصوت المصدر أعلاه وإيقافه وإرجاعه.',
    sourceText: 'المصدر',
    notesPlaceholder: 'دوّن ملاحظاتك هنا أثناء الاستماع (رموز، أسهم، كلمات مفتاحية)…',
    notesHint: 'تبقى ملاحظاتك في هذه الصفحة فقط — لا تُحفظ ولا تُرسل إلى أي مكان. لوحة الرسم تدعم القلم الرقمي (حساس للضغط).',
    simulTitle: '🎧 الوضع الفوري',
    simulStart: '▶ تشغيل المصدر + تسجيلي',
    simulStop: '⏹ إيقاف الاثنين',
    simulHint: 'استخدم سماعات رأس حتى لا يلتقط الميكروفون صوت المصدر. يُشغَّل الخطاب المصدر بينما تُسجَّل ترجمتك — تدريب فوري حقيقي.',
    simulNeedsAudio: 'ولّد الصوت المصدر أولاً في «الصوت والمواد» لتفعيل الوضع الفوري.',
    prepTimeLabel: 'وقت التحضير قبل البدء بالكلام — لا يُحتسب ضد الطلاقة',
    fluencyExplain: 'تُقاس هذه الدرجة من تسجيلك: سرعة الإلقاء، والتوقّفات أثناء الأداء، ومدى وضوح كل كلمة عند التعرّف عليها. أمّا الوقت الذي تستغرقه في التحضير قبل الكلام فلا يُحتسب عليك. والكلمة التي لم يتبيّنها النظام بوضوح تُعدّ مؤشّراً على وضوح الصوت، لا خطأً في الترجمة.',
    aboutEvalTitle: 'ℹ️ كيف يعمل هذا التقييم وأين تذهب بياناتك',
    aboutEvalCriteria: 'ما يُحلَّل: (1) يُفرَّغ تسجيلك بواسطة نظام التعرف على الكلام (Whisper)؛ (2) تُرصد الأحداث القابلة للقياس مباشرة من الصوت — التوقفات، التكرارات، التردد، سرعة الكلام؛ (3) يقارن مقيّم ذكاء اصطناعي ترجمتك بالخطاب المصدر معنىً بمعنى: الدقة، التغطية، المصطلحات، الأرقام والتواريخ، القواعد؛ (4) تتبع الدرجات معيار أستاذ صارم (تقع معظم أداءات الطلبة بين 4 و7 من 10).',
    aboutEvalReliability: 'الموثوقية: النص المفرَّغ تقدير وليس حقيقة مطلقة — خصوصاً للعربية. وتُعامل المقاطع غير الواضحة على أنها مشكلات في وضوح الصوت، لا أخطاءَ ترجمة. استخدم هذا التقرير كأداة تدريب، لا كعلامة نهائية.',
    aboutEvalStorage: 'التخزين: يُعالَج تسجيلك الصوتي ثم يُحذف من الخادم ولا يُحتفظ به. وحدَه ملخّص التقييم يُحفظ في سجلّ حسابك، ويتضمّن العلامات وعدد الأخطاء والتوصيات. لا تُخزَّن النصوص المولَّدة والمفرَّغة بشكل دائم، لكن يمكن نسخها أو تحميلها.',
    creditsLine: 'ETIB · ESIB · CINIA · جامعة القديس يوسف بيروت',
    modeChoose: 'وضع التدريب',
    modeSimulDesc: 'يُشغَّل الخطاب المصدر بينما تترجم وتسجّل في الوقت نفسه.',
    modeConsecDesc: 'استمع إلى المصدر ودوّن ملاحظاتك، ثم سجّل ترجمتك.',
    modeSightDesc: 'اقرأ النص المتحرك وترجم بصوت عالٍ أثناء تسجيل نفسك.',
    heroTrainIn: 'تدرّب بـ:',
    heroTitlePre: 'أتقن الترجمة الشفهية مع ',
    heroTitleEm: 'خطابات مولّدة بالذكاء الاصطناعي',
    heroTitlePost: '',
    heroParagraph: 'منصة تدريب تكيفية تولّد خطابات مؤتمرات واقعية، وتبني مسارد متعددة اللغات، وتقيّم أداءك في الترجمة الشفهية في العربية والفرنسية والإنجليزية.',
    guestBtn: 'المتابعة كضيف',
    orSep: '— أو —',
    srcPanelTitle: 'إضافة مصدر',
    srcPanelSubtitle: 'ابحث عن وثيقة أممية أو صفحة ويب أو ارفع ملفك الخاص لتأسيس الخطاب على محتوى حقيقي.',
    srcTabUN: 'مكتبة الأمم المتحدة',
    srcTabUpload: 'رفع ملف',
    srcUNHint: 'ابحث في مكتبة الأمم المتحدة الرقمية عن موضوعك، نزّل ملف PDF، ثم ارفعه أدناه.',
    srcUNSearchBtn: '↗ البحث في مكتبة الأمم المتحدة',
    srcUNUploadAfter: 'بعد تنزيل ملف PDF، ارفعه هنا:',
    srcDropPdf: 'أسقط ملف PDF هنا أو',
    srcDropFile: 'أسقط ملفاً هنا أو',
    srcBrowse: 'تصفّح',
    srcFileTypes: 'PDF أو Word (.docx) أو نص عادي',
    audioFluencyTitle: 'الطلاقة مقيسةً من التسجيل',
    coverageScoreLabel: 'درجة التغطية',
    statSpeechRate: 'سرعة الكلام',
    statLongPauses: 'توقفات طويلة',
    statSilenceRatio: 'نسبة الصمت',
    statWordConfidence: 'وضوح الكلمات',
    verdictGood: 'جيد',
    verdictAcceptable: 'مقبول',
    verdictNeedsWork: 'يحتاج إلى تحسين',
    nextHintPre: '← انتقل إلى',
    nextHintPost: 'للاستماع ومراجعة الأسئلة وتنزيل المسرد.',
    transcriptReady: 'النص المفرَّغ جاهز',
    chatGreeting: 'مرحباً! أنا مساعد ETIB. اسألني عن تقنيات الترجمة الفورية أو المنصة أو المصطلحات.',
    chatPlaceholder: 'اسأل أي شيء…',
    chatSend: 'إرسال',
    chatThinking: 'أفكر…',
    statAvgPerSession: 'متوسط / جلسة',
    statTotal: 'الإجمالي',
    statSessions: 'الجلسات',
    outsideRange: '(خارج النطاق المطلوب)',
    progressGuestNote: 'يعمل تتبع التقدم مع حساب فقط. سجّل الدخول (أو أنشئ حساباً مجانياً) لتُحفظ تقييماتك وتُحلَّل عبر الجلسات.',
    pnTitle: 'أسماء الأعلام (أشخاص، منظمات، أماكن)',
    pnCorrect: 'صحيح',
    pnDistorted: 'محرَّف — نطق خاطئ على الأرجح',
    pnMissing: 'محذوف',
    llmFailedWarning: 'لم يتمكن التحليل اللغوي بالذكاء الاصطناعي من الاكتمال — يُعرض الرصد التلقائي فقط. قد تكون الدرجات غير مكتملة.',
    llmQuotaError: 'بلغت خدمة الذكاء الاصطناعي حدّها في الدقيقة — قد يحدث ذلك مع الخطابات الطويلة أو عند إرسال عدة طلبات متتالية. أعادت المنصة المحاولة تلقائياً؛ انتظر نحو دقيقة ثم أعد المحاولة. يُعاد ضبط الحدّ المشترك تلقائياً — لا حاجة لأي إعداد.',
    keyInvalid: 'رفضت خدمة الذكاء الاصطناعي الطلب (مفتاح الخادم غير صالح أو منتهٍ). ليس بإمكانك إصلاح ذلك — أبلغ أستاذك أو مسؤول المنصّة.',
    sameLangError: 'يجب أن تختلف لغة المصدر عن لغة الترجمة الشفوية — فالترجمة الفورية تكون دائماً بين لغتين مختلفتين.',
    simulSourceEnded: 'انتهى الخطاب المصدر — أكمل ترجمتك ثم اضغط إيقاف. يتوقف التسجيل تلقائياً بعد دقيقة واحدة.',
    simulWithText: 'إظهار نص المصدر على الشاشة (فورية مع النص)',
    consecFlowHint: 'الخطوة ١: استمع إلى الخطاب ودوّن ملاحظاتك. الخطوة ٢: سجّل ترجمتك اعتماداً على ملاحظاتك أدناه.',
    simulUseButton: 'استخدم زر «تشغيل المصدر والتسجيل» أعلاه للبدء.',
    sightUseButton: 'استخدم زر التسجيل بجانب النص المتحرك أعلاه للبدء.',
    downloadRecording: 'تنزيل التسجيل',
    downloadAudio: 'تنزيل الصوت',
    llmDownError: 'لم تستجب خدمة التقييم بالذكاء الاصطناعي — حاول مرة أخرى بعد قليل.',
    addSource: 'إضافة مصدر',
    uploadSourceMedia: 'رفع خطاب للترجمة',
    sourceMediaEmpty: 'تعذّر تفريغ الوسائط المرفوعة — جرّب تسجيلاً أوضح.',
    micDenied: 'تم رفض الوصول إلى الميكروفون — يرجى السماح بالإذن والمحاولة مجدداً.',
    simultaneousFailed: 'تعذّر تشغيل الوضع الفوري — تحقق من إذن الميكروفون.',
    topErrorSaid: 'ما قيل',
    topErrorExpected: 'المتوقع',
    topErrorMissing: 'مفقود',
    uploadAudioBtn: 'رفع ملف صوتي',
    correctTashkeel: 'تشكيل صحيح',
    iraabTitle: 'أواخر الكلمات (الإعراب)',
    iraabAcousticNote: 'تُرصَد أواخر الكلمات صوتيًّا من تسجيلك؛ ويبقى النص المفرَّغ حرفيًّا كما نطقتَه، فلا تُصحَّح الأواخر الخاطئة.',
    iraabDetected: 'المسموع',
    iraabCorrect: 'صحيح',
    iraabIncorrect: 'خطأ',
    iraabUncertain: 'غير مؤكَّد',
    iraabAccuracy: 'دقة الإعراب',
    iraabDiacritized: 'النص المشكَّل (فرضية)',
  },
  fr: {
    uiLanguage: 'Langue de l’interface',
    loginTitle: 'Connexion à ETIB Trainer',
    loginSubtitle: 'Accédez à votre espace d’entraînement à l’interprétation.',
    signupTitle: 'Créer votre compte ETIB Trainer',
    signupSubtitle: 'Créez un compte pour commencer l’entraînement.',
    name: 'Nom complet',
    email: 'E-mail',
    password: 'Mot de passe',
    showPassword: 'Afficher',
    hidePassword: 'Masquer',
    role: 'Rôle',
    loginSubmit: 'Se connecter',
    loginLoading: 'Connexion...',
    signupSubmit: 'Créer le compte',
    signupLoading: 'Création du compte...',
    loginNote: 'Utilisez vos identifiants ou créez un nouveau compte.',
    switchToSignup: 'Créer un compte',
    switchToLogin: 'Vous avez déjà un compte ? Se connecter',
    student: 'Étudiant',
    guestName: 'Invité',
    instructor: 'Formateur',
    coordinator: 'Coordinateur',
    signOut: 'Se déconnecter',
    workspaceKicker: 'Modules d’entraînement',
    workspaceTitle: 'Générez des discours, préparez les supports, enregistrez les interprétations et consultez les retours.',
    navA: 'Générer un discours',
    navB: 'Audio et supports',
    navC: 'Enregistrer et transcrire',
    navD: 'Évaluation',
    navE: 'Progression',
    progressTitle: 'Ma progression',
    progressSubtitle: 'Historique des scores et recommandations adaptatives',
    progressNoSessions: 'Aucune session — terminez une évaluation complète pour commencer le suivi.',
    progressSessions: 'Historique des sessions',
    progressAdaptive: 'Recommandations adaptatives',
    progressAdaptiveSubtitle: 'D\’après vos dernières sessions, la plateforme recommande :',
    progressApply: 'Appliquer ces paramètres',
    progressOverall: 'Global',
    progressFluency: 'Fluidité',
    progressCoverage: 'Couverture',
    progressErrors: 'Erreurs',
    progressLang: 'Langue',
    progressDate: 'Date',
    progressScore: 'Score',
    progressLoading: 'Chargement…',
    progressAvgOver: 'Moy. globale',
    progressAvgFluency: 'Moy. fluidité',
    progressAvgNumbers: 'Moy. erreurs numériques',
    progressTrend: 'Courbe des scores (10 dernières sessions)',
    progressLatest: 'Dernière session',
    progressProblems: 'Points de travail — axes prioritaires',
    progressImproving: 'En progression',
    progressDeclining: 'En recul',
    progressStable: 'Stable',
    progressStrengths: 'Points forts',
    progressRecs: 'Recommandations',
    severityHigh: 'HAUTE', severityMedium: 'MOY.', severityLow: 'FAIBLE',
    problemLabel_translation: 'Précision de la traduction',
    problemLabel_numbers: 'Chiffres et statistiques',
    problemLabel_missing: 'Contenu manquant',
    problemLabel_silences: 'Longues pauses',
    problemLabel_repetitions: 'Répétitions et autocorrections',
    problemLabel_fluency: 'Fluidité',
    problemLabel_coverage: 'Couverture du contenu',
    problemTip_translation: "Révisez les listes de terminologie et pratiquez l'écoute active.",
    problemTip_numbers: 'Entraînez-vous sur des discours à haute densité de chiffres.',
    problemTip_missing: "Améliorez votre prise de notes et travaillez d'abord sur des discours courts.",
    problemTip_silences: 'Pratiquez le décalage pour maintenir le flux de restitution.',
    problemTip_repetitions: 'Travaillez la reformulation — engagez-vous dans un rendu sans revenir en arrière.',
    problemTip_fluency: 'Faites du shadowing avec des locuteurs natifs et enregistrez-vous chaque jour.',
    problemTip_coverage: 'Concentrez-vous sur les idées principales avant de vous soucier du style.',
    problemDetail_translation: 'Équivalents erronés dans {count}/{total} sessions',
    problemDetail_numbers: 'Erreurs numériques dans {count}/{total} sessions',
    problemDetail_missing: 'Idées clés omises dans {count}/{total} sessions',
    problemDetail_silences: 'Pauses excessives dans {count}/{total} sessions',
    problemDetail_repetitions: 'Répétitions élevées dans {count}/{total} sessions',
    problemDetail_fluency: 'Fluidité faible dans {count}/{total} sessions',
    problemDetail_coverage: 'Couverture faible dans {count}/{total} sessions',
    progressTopErrors: 'Erreurs spécifiques',
    progressNoData: 'Pas encore assez de données',
    moduleATitle: 'Générer un discours d’entraînement',
    groundedSourceLabel: 'Basé sur une source réelle :',
    ungroundedNote: 'Aucun document de l’ONU correspondant n’a été trouvé — ce discours a été généré sans source.',
    language: 'Langue du discours',
    targetLanguage: 'Langue cible',
    topic: 'Sujet',
    domain: 'Domaine',
    speedPressure: 'Pression de vitesse',
    topicShifts: 'Changements de sujet',
    pressureMode: 'Mode pression',
    contextNoise: 'Bruit de contexte',
    cognitiveLoad: 'Charge cognitive',
    summary: 'Résumé',
    mcq: 'QCM',
    glossary: 'Glossaire',
    documentGrounding: 'Génération depuis un document',
    documentFile: 'Document source',
    sharedSettings: 'Paramètres de génération',
    sharedSettingsHint: 'Ces paramètres s\'appliquent à la génération par sujet ou par document.',
    generationMethods: 'Choisir la source de génération',
    topicOnlyGeneration: 'Génération par sujet',
    topicOnlyHint: 'Générer à partir du sujet et des paramètres choisis.',
    documentGenerationHint: 'Déposez un document source pour générer un discours ancré dans son contenu.',
    generateFromDocument: 'Générer depuis le document',
    generationEmpty: 'Aucun texte de discours n\'a été généré (le serveur se réveille peut-être, ou la requête a été limitée). Rien n\'a été conservé — veuillez réessayer.',
    retrieveContext: 'Prévisualiser le contexte extrait',
    retrievedContext: 'Contexte extrait',
    wordCount: 'Longueur du discours',
    wordRangeShort: 'Court',
    wordRangeMedium: 'Moyen',
    wordRangeLong: 'Long',
    difficulty: 'Difficulté',
    structure: 'Structure du discours',
    mode: 'Mode d’interprétation',
    numbers: 'Densité des chiffres',
    hesitationsSim: 'Simuler les hésitations',
    submit: 'Générer le discours',
    generating: 'Génération en cours...',
    wordsUnit: 'mots',
    moduleBTitle: 'Audio et supports pédagogiques',
    moduleBBody: 'Produira : lecture audio, termes clés, glossaire et QCM.',
    moduleCTitle: 'Enregistrer et transcrire',
    moduleCBody: 'Permettra : enregistrement dans le navigateur, transcription ASR, horodatage et lecture.',
    moduleDTitle: 'Évaluation des performances',
    moduleDBody: 'Détectera : hésitations, omissions, erreurs de chiffres, problèmes terminologiques et faiblesses récurrentes.',
    comingSoon: 'Bientôt disponible',
    errorPrefix: 'Erreur',
    noSpeechYet: 'Générez d\'abord un discours dans le Module A, puis revenez ici.',
    materialsMissing: 'Les questions de compréhension et le glossaire de ce discours sont absents (ils peuvent disparaître sur les discours très longs). Générez-les maintenant :',
    materialsGenerating: 'Génération…',
    voiceAccent: 'Accent vocal',
    speechRate: 'Débit de parole',
    sessionRestored: 'Votre session précédente a été restaurée (discours, supports et modifications du glossaire). Vous pouvez reprendre là où vous étiez.',
    sessionKeep: 'Continuer',
    sessionDiscard: 'Recommencer à zéro',
    measuredRate: 'Débit mesuré',
    wordsPerMinute: 'mots/min',
    rateTooFastHint: 'au-dessus de la plage 100–120 mots/min recommandée pour la simultanée — réduisez le débit avec le curseur puis régénérez',
    mediaUnreadable: 'Ce fichier n\'a pas pu être lu. Formats pris en charge : MP3, WAV, M4A, OGG, WebM, MP4. S\'il fonctionne sur votre appareil, convertissez-le en MP3 ou WAV puis réessayez.',
    generateAudio: 'Générer l\'audio',
    generatingAudio: 'Génération audio...',
    materialsTitle: 'Supports pédagogiques',
    generateMaterials: 'Générer les supports',
    generatingMaterials: 'Génération des supports...',
    materialsFailed: 'Le discours a été transcrit, mais les supports (résumé, QCM, glossaire) n’ont pas pu être générés — le serveur se réveille peut-être (2–3 min). Cliquez à nouveau sur « Importer un discours » pour réessayer.',
    keyTerms: 'Termes clés',
    summaryTitle: 'Résumé thématique',
    mcqTitle: 'Questions de compréhension (QCM)',
    mcqQuestionHeader: 'Question',
    mcqOptionsHeader: 'Options',
    mcqAnswerHeader: 'Réponse',
    glossaryTitle: 'Glossaire trilingue (AR – FR – EN)',
    glossaryDuringRecording: 'Glossaire (gardez la terminologie visible pendant l’interprétation)',
    glossaryTermHeader: 'Terme',
    glossaryArabicHeader: 'Arabe',
    glossaryFrenchHeader: 'Français',
    glossaryEnglishHeader: 'Anglais',
    glossaryDefinitionHeader: 'Définition',
    downloadGlossary: 'Télécharger le glossaire (DOCX)',
    uploadGlossary: 'Importer un glossaire',
    glossaryUploading: 'Importation…',
    glossaryUploadEmpty: 'Aucun terme n\'a pu être lu dans ce fichier. Utilisez des colonnes comme Terme / Arabe / Français / Anglais (CSV, TSV, TXT, DOCX ou JSON).',
    uploadAudio: 'Déposez votre interprétation (MP3, WAV, M4A)',
    transcribeBtn: 'Transcrire',
    transcribing: 'Transcription en cours — peut prendre 1–2 min...',
    transcriptLabel: 'Transcription',
    lowConfidence: 'Mots mal perçus par la reconnaissance (à vérifier)',
    arabicTip: 'Pour de meilleurs résultats : parlez clairement en arabe standard (فصحى) dans une pièce calme.',
    noTranscriptYet: 'Enregistrez d\'abord une interprétation dans "Enregistrer et transcrire", puis revenez ici.',
    runEvaluation: 'Lancer l\'évaluation complète',
    runningEvaluation: 'Analyse en cours...',
    overallScore: 'Score global',
    evalSummary: 'Évaluation globale',
    strengths: 'Points forts',
    recommendations: 'Recommandations',
    algorithmicTitle: 'Détections automatiques',
    longSilences: 'Longs silences',
    repetitions: 'Répétitions',
    hesitations: 'Hésitations',
    filledPause: 'pause remplie (euh…)',
    cutWord: 'mot coupé',
    numberErrors: 'Erreurs de chiffres',
    llmTitle: 'Analyse linguistique IA',
    languageErrors: 'Erreurs linguistiques',
    autoCorrections: 'Auto-corrections',
    falseStarts: 'Faux départs',
    lapsusLinguae: 'Lapsus linguae',
    terminologyProblems: 'Problèmes terminologiques',
    informationLoss: 'Perte d\'information',
    noIssues: 'Aucun détecté',
    correction: 'Correction',
    source: 'Source',
    used: 'Utilisé',
    correct: 'Terme correct',
    coverageTitle: 'Couverture du contenu',
    coverageHint: 'Quantité du discours source transmise',
    translationErrors: 'Erreurs de traduction',
    meaningCheckTitle: 'Vérification du sens',
    meaningCheckNote: 'Changements de sens possibles — affichés pour vous aider à progresser ; ils n\'affectent PAS votre note. La même idée peut être formulée différemment, donc seuls les véritables changements de sens sont signalés.',
    sourceSaid: 'Source a dit',
    studentSaid: "L'étudiant a dit",
    correctTranslation: 'Traduction correcte',
    missingContent: 'Contenu manquant',
    scoreScopeNote: 'Ce score reflète la conformité terminologique, la correction de la langue cible, les chiffres/noms et la restitution technique (pauses, gestion du décalage, fluidité). Le sens/la couverture ci-dessous sont affichés à titre informatif et n\'influencent pas le score à ce stade.',
    aiDisclaimerTitle: '⚠️ Avertissement IA',
    aiDisclaimerShort: 'L\'IA peut se tromper — vérifiez le texte, l\'audio, la transcription et les scores avant de vous y fier.',
    aiDisclaimerMore: 'Détails',
    aiDisclaimerLess: 'Fermer',
    aiDisclaimerGen: "Discours générés : Le texte est produit par un grand modèle de langage. Il peut contenir des chiffres inventés, des noms de rapports ou de résolutions inexistants, des erreurs factuelles et des biais culturels ou politiques. Ne le considérez jamais comme une source d'information : c'est un support d'entraînement. En arabe, l'accord du nombre (العدد وأحكامه), les signes diacritiques (التحريك) et les dates sont les éléments qui donnent lieu au plus grand nombre d'erreurs.",
    aiDisclaimerTts: "Audio synthétique (TTS) : Les noms, sigles, dates et chiffres risquent d'être mal prononcés — surtout en arabe. Ce que vous entendez ne correspond pas toujours au texte écrit. Modifiez le texte avant de générer l'audio si un passage semble incorrect.",
    aiDisclaimerStt: "Transcription (STT) : Votre enregistrement est transcrit automatiquement. La transcription est une estimation et peut ne pas correspondre exactement à ce que vous avez dit. L'usage de la langue arabe, l'alternance codique et les bruits de fond en dégradent nettement la précision. Corrigez la transcription avant de lancer l'évaluation.",
    aiDisclaimerEval: "Évaluation : Le rapport, généré automatiquement, est un outil d'aide à l'apprentissage. Il ne remplace ni une note ni l'enseignant, et n'évalue que les catégories précisées ci-dessous ; le reste des informations ou observations n'est donné qu'à titre indicatif.",
    aiDisclaimerAdvice: 'Il est recommandé de vérifier la terminologie dans les bases de données de référence ainsi que les chiffres et les noms propres en les comparant à la source, de corriger la transcription et de signaler à votre enseignant toute information qui vous semble incorrecte.',
    aiDisclaimerAck: 'J\'ai compris',
    ownTextBtn: '\u270d\ufe0f Utiliser mon propre texte (sans IA)',
    ownTextTitle: '\u270d\ufe0f Mon propre texte \u2014 audio seulement, sans g\u00e9n\u00e9ration IA',
    ownTextHint: 'Collez un texte que vous avez d\u00e9j\u00e0. Il est utilis\u00e9 EXACTEMENT tel quel : aucune r\u00e9\u00e9criture, aucune g\u00e9n\u00e9ration par l\'IA. Passez ensuite \u00e0 \u00ab Audio & supports \u00bb pour le faire lire, puis interpr\u00e9tez-le normalement.',
    ownTextPlaceholder: 'Collez votre texte ici\u2026',
    ownTextUse: 'Utiliser ce texte tel quel',
    ownTextEmpty: 'Collez d\'abord un texte.',
    ownTextBadge: 'Votre propre texte \u2014 non g\u00e9n\u00e9r\u00e9 par IA',
    ownTextMaterialsHint: 'Ce texte a \u00e9t\u00e9 ajout\u00e9 tel quel : il n\'a donc ni r\u00e9sum\u00e9, ni QCM, ni glossaire. G\u00e9n\u00e9rez-les avec le bouton si vous les souhaitez \u2014 le texte lui-m\u00eame n\'est jamais modifi\u00e9.',
    scriptTitle: 'Texte du discours',
    editScriptBtn: '\u270e Modifier le texte',
    editScriptDone: '\u2713 Termin\u00e9',
    editScriptHint: 'Corrigez le texte AVANT de g\u00e9n\u00e9rer l\'audio \u2014 erreurs, chiffres mal formul\u00e9s, noms erron\u00e9s. L\'audio, votre interpr\u00e9tation et l\'\u00e9valuation utilisent tous cette version corrig\u00e9e.',
    scriptRestore: '\u21ba R\u00e9tablir l\'original',
    editTranscriptBtn: '\u270e Corriger la transcription',
    editTranscriptDone: '\u2713 Enregistrer la correction',
    editTranscriptHint: 'La transcription est ce que la reconnaissance vocale a entendu. Corrigez toute erreur de reconnaissance AVANT l\'\u00e9valuation, pour qu\'une erreur de la machine ne soit pas compt\u00e9e comme la v\u00f4tre. Les pauses, h\u00e9sitations et le d\u00e9bit restent mesur\u00e9s sur votre enregistrement r\u00e9el.',
    transcriptEditedBadge: 'transcription corrig\u00e9e par vous',
    micWarmingUp: 'Pr\u00e9paration du microphone\u2026',
    micReadyHint: 'Attendez l\'affichage \u00ab Enregistrement \u00bb avant de commencer \u00e0 parler.',
    mediaAsrWarning: 'Ce texte provient de la transcription automatique de votre fichier. Sa pr\u00e9cision d\u00e9pend de la qualit\u00e9 audio (et reste plus faible en arabe) : relisez-le et corrigez-le dans \u00ab Audio & supports \u00bb avant de l\'utiliser comme source.',
    fluencyScore: 'Score de fluidité',
    pauseAnalysisTitle: 'Pauses et rythme',
    repetitionAnalysisTitle: 'Répétitions — analyse',
    numberAccuracyTitle: 'Chiffres et dates',
    pronunciationAssessmentTitle: 'Retour sur la prononciation',
    numCorrect: 'Correct',
    numIncorrect: 'Incorrect',
    expectedLabel: 'Attendu',
    studentSaidShort: 'Dit',
    pronunciationTitle: 'Évaluation de la prononciation',
    pronunciationMismatches: 'Divergences de clarté mot par mot',
    runPronunciation: 'Lancer le contrôle de prononciation',
    runningPronunciation: 'Alignement audio en cours — peut prendre 1 min...',
    pronunciationScore: 'Certitude de détection',
    wordView: 'Vue mot par mot',
    legend: 'Légende',
    legendGood: '≥80% clair',
    legendWarn: '60–80%',
    legendPoor: '<60% — erreur probable',
    uncertainWords: 'Mots importants peu clairs',
    expectedForm: 'Forme attendue',
    likelyError: 'Erreur probable',
    grammaticalRole: 'Rôle grammatical',
    noUncertain: "Aucun mot important n'a semblé peu clair",
    whisperxBadge: 'WhisperX aligné',
    fallbackBadge: 'Scores Whisper',
    needsArabic: 'Le contrôle de prononciation est disponible pour l\'arabe uniquement.',
    needsSource: 'Générez d\'abord un discours dans le Module A.',
    sourceAudio: 'Audio du discours source',
    playbackSpeed: 'Vitesse de lecture',
    yourInterpretation: 'Votre interprétation',
    recordBtn: '🎙 Démarrer l\'enregistrement',
    stopBtn: '⏹ Arrêter l\'enregistrement',
    recordingLive: 'Enregistrement en cours',
    autoTranscribe: 'Transcrire automatiquement après l\'enregistrement',
    orUpload: 'Ou déposez un fichier audio',
    deleteAudio: 'Supprimer l\'audio',
    redoTranscript: 'Refaire la transcription',
    copyText: 'Copier',
    bubbleGood: 'Bonne transcription',
    bubblePoor: 'Mauvaise transcription',
    deleteTranscript: 'Supprimer la transcription',
    closeLabel: 'Fermer',
    copied: 'Copié !',
    topicPlaceholder: 'Saisissez un sujet ou collez un texte pour générer un discours…',
    namePlaceholder: 'Votre nom complet',
    librarySearchPlaceholder: "ex. changement climatique, gouvernance de l'IA…",
    removeSource: 'Retirer la source',
    addSourceFirst: "Ajoutez d'abord au moins un document source.",
    moreSettings: 'Plus ⚙',
    lessSettings: 'Moins ⚙',
    optConsecutive: 'Consécutive',
    optSimultaneous: 'Simultanée',
    optSight: 'Traduction à vue',
    optWellOrganized: 'Bien structuré',
    optDisorganized: 'Peu structuré',
    optLowNumbers: 'Peu de chiffres',
    optHighNumbers: 'Beaucoup de chiffres',
    optNormalSpeed: 'Vitesse normale',
    optFast: 'Rapide',
    optVeryFast: 'Très rapide',
    optNoShifts: 'Sans changements',
    optShifts: 'Changements de sujet',
    domPolitics: 'Politique',
    domDiplomacy: 'Diplomatie',
    domEconomics: 'Économie',
    domHealth: 'Santé',
    domEducation: 'Éducation',
    domClimate: 'Climat & Environnement',
    domHumanRights: 'Droits de l\'homme',
    domTechnology: 'Technologie & IA',
    domMigration: 'Migration & Réfugiés',
    domDisarmament: 'Désarmement',
    domWomen: 'Femmes & Genre',
    domFood: 'Alimentation & Faim',
    domLegal: 'Juridique',
    domMedical: 'Médical',
    diffBeginner: 'Débutant',
    diffIntermediate: 'Intermédiaire',
    diffAdvanced: 'Avancé',
    diffHint: 'Débutant : vocabulaire simple, phrases courtes, 1–2 chiffres, débit lent. Intermédiaire : terminologie modérée, plusieurs statistiques, phrases variées. Avancé : terminologie dense, chiffres et noms fréquents, syntaxe complexe, débit rapide.',
    wordRangeShortFull: 'Court — 120–180 mots (≈ 1 min)',
    wordRangeMediumFull: 'Moyen — 220–320 mots (≈ 1,5–2,5 min)',
    wordRangeLongFull: 'Long — 400–550 mots (≈ 3–4 min)',
    wordRangeExtendedFull: 'Étendu — 650–800 mots (≈ 4,5–6 min)',
    wordRangeVeryLongFull: 'Très long — 1100–1400 mots (≈ 8–11 min)',
    wordRangeExtraLongFull: 'Très très long — 1500–2200 mots (≈ 12–18 min)',
    wordRangeMarathonFull: 'Marathon — 2900–3600 mots (≈ 24–30 min)',
    optSemiStructured: 'Semi-structuré',
    optDisorganizedFull: 'Volontairement désorganisé',
    scenarioLabel: 'Style d\'orateur / contexte',
    scenUNGA: 'Assemblée générale de l\'ONU',
    scenEUParl: 'Parlement européen',
    scenArabLeague: 'Sommet de la Ligue arabe',
    scenPress: 'Conférence de presse',
    scenDiplomatic: 'Réunion diplomatique',
    scenDebate: 'Débat politique',
    scenInterview: 'Entrevue',
    scenGroupGeneral: 'Général / conférence',
    scenGroupSimul: 'Orienté simultanée',
    scenGroupConsec: 'Interprétation de service public',
    scenGroupSight: 'Orienté traduction à vue',
    scenGroupInstitutions: 'Institutions internationales',
    scenPanel: 'Table ronde',
    scenLiveTV: 'Diffusion TV en direct',
    scenLegalCourt: 'Juridique / Tribunal (procès, police, déposition)',
    scenMedical: 'Médical / Santé (clinique, hôpital, thérapie)',
    scenPublicService: 'Service public (consultation juridique ou médicale)',
    scenSecurityCouncil: 'Conseil de sécurité de l\'ONU',
    scenEcosoc: 'ECOSOC (Conseil économique et social)',
    scenWho: 'OMS (Organisation mondiale de la Santé)',
    scenIlo: 'OIT (Organisation internationale du Travail)',
    scenUnesco: 'UNESCO',
    scenOif: 'OIF (Francophonie)',
    scenAuf: 'AUF (universités francophones)',
    termDensity: 'Densité terminologique',
    optTermLow: 'Faible — vocabulaire courant',
    optTermMedium: 'Moyenne',
    optTermHigh: 'Élevée — termes spécialisés denses',
    longTextAsSource: 'Texte long détecté — il sera utilisé comme document source et le discours s\'appuiera sur son contenu.',
    mcqPickAnswer: 'Choisissez votre réponse, puis vérifiez-la.',
    mcqCheck: 'Vérifier la réponse',
    mcqShowAnswer: 'Afficher la réponse',
    mcqCorrectMsg: '✓ Correct !',
    mcqWrongMsg: '✗ Incorrect. Bonne réponse :',
    mcqShowAnswerLabel: 'ℹ️ Bonne réponse :',
    mcqYourScore: 'Votre score',
    glossaryEdit: '✏️ Modifier le glossaire',
    glossaryAddTerm: 'Ajouter un terme',
    glossaryEditDone: '✓ Terminer la modification',
    glossaryEditHint: 'Les équivalents sont recherchés dans les bases officielles (Thesaurus UNBIS, IATE, FranceTerme) — un ✓ signale un terme qui y est confirmé. Relisez-les et corrigez-les AVANT d\'enregistrer : l\'évaluation contrôle votre terminologie par rapport à ce glossaire.',
    termVerifiedTitle: 'Confirmé dans',
    termEditedTitle: 'votre correction',
    glossaryMemoryNote: 'corrections mémorisées — elles sont réappliquées automatiquement aux prochains glossaires.',
    glossaryMemoryClear: 'Oublier les corrections enregistrées',
    webPageTab: 'Page web',
    webPageUrlPlaceholder: 'https://… collez l\'adresse d\'un article ou d\'un rapport',
    webPageFetch: 'Récupérer la page',
    webPageFetching: 'Récupération…',
    webPageAttached: 'Source ajoutée',
    webPageDone: 'Terminé',
    fileTypeError: 'Type de fichier non pris en charge. Importez un PDF, un Word (.docx) ou un fichier texte (.txt).',
    webPageHint: 'Le texte lisible de la page sera extrait et utilisé comme source du discours.',
    scrollerTitle: 'Traduction à vue — texte défilant',
    scrollerPlay: '▶ Lancer le défilement',
    scrollerPause: '⏸ Pause',
    scrollerReset: '↺ Réinitialiser',
    scrollerSpeed: 'Vitesse (mots/min)',
    scrollerFontSize: 'Taille du texte',
    scrollerColumns: 'Colonnes',
    scrollerSpacing: 'Interligne',
    scrollerHint: 'Traduisez à voix haute pendant que le texte défile à la vitesse choisie — comme un vrai exercice de traduction à vue.',
    notesTitle: '📝 Prise de notes (consécutive)',
    notesTextTab: 'Notes texte',
    notesSketchTab: 'Croquis / carte mentale',
    notesClear: 'Effacer',
    notesShowSource: 'Afficher la source',
    notesHideSource: 'Masquer la source',
    liveTranscriptOn: 'Transcription en direct',
    liveTranscriptOff: 'Masquer la transcription',
    liveTranscriptTitle: 'Transcription en direct',
    liveTranscriptWaiting: 'Lancez l\'audio source — la transcription apparaîtra ici au fil de la lecture.',
    liveTranscriptHint: 'Écran partagé : le discours source est transcrit d\'un côté pendant sa lecture ; prenez vos notes de l\'autre. Contrôlez le tout avec la lecture/pause/retour de l\'audio source ci-dessus.',
    sourceText: 'Source',
    notesPlaceholder: 'Prenez vos notes ici pendant l\'écoute (symboles, flèches, mots-clés)…',
    notesHint: 'Vos notes restent sur cette page uniquement — elles ne sont ni enregistrées ni envoyées. Le croquis prend en charge un stylet (sensible à la pression).',
    simulTitle: '🎧 Mode simultané',
    simulStart: '▶ Lire la source + m\'enregistrer',
    simulStop: '⏹ Tout arrêter',
    simulHint: 'Portez un casque pour que l\'audio source ne soit pas capté par votre micro. Le discours source est lu pendant que votre interprétation est enregistrée — un vrai entraînement simultané.',
    simulNeedsAudio: 'Générez d\'abord l\'audio source dans « Audio et supports » pour activer le mode simultané.',
    prepTimeLabel: 'temps de préparation avant de parler — non compté dans la fluidité',
    fluencyExplain: 'Ce score est mesuré à partir de votre enregistrement : débit de parole, pauses pendant la prestation et clarté de reconnaissance de chaque mot. Le temps de préparation avant de commencer n\'est PAS compté. Un mot mal perçu par la reconnaissance signifie qu\'il a semblé peu clair — c\'est un indice de clarté, pas automatiquement une erreur.',
    aboutEvalTitle: 'ℹ️ Comment fonctionne cette évaluation et où vont vos données',
    aboutEvalCriteria: 'Ce qui est analysé : (1) votre enregistrement est transcrit par une reconnaissance vocale (Whisper) ; (2) les événements mesurables sont détectés directement dans l\'audio — pauses, répétitions, hésitations, débit ; (3) un évaluateur IA compare votre interprétation au discours source sens par sens : exactitude, couverture, terminologie, chiffres et dates, grammaire ; (4) les scores suivent un barème de professeur strict (la plupart des prestations étudiantes obtiennent 4–7/10).',
    aboutEvalReliability: 'Fiabilité : la transcription est une estimation, pas une vérité absolue — surtout en arabe. Les passages peu clairs sont traités comme des problèmes de clarté, pas automatiquement comme des erreurs de traduction. Utilisez ce rapport comme outil d\'entraînement, pas comme note finale.',
    aboutEvalStorage: "Stockage : Votre enregistrement audio est traité puis supprimé du serveur ; il n'est pas conservé. Seul le récapitulatif de l'évaluation (scores, nombre d'erreurs et recommandations) est enregistré dans l'historique de votre compte. Les discours et les transcriptions générés ne sont pas conservés de manière permanente, mais il est possible de les copier ou de les télécharger.",
    creditsLine: 'ETIB · ESIB · CINIA · USJ Beyrouth',
    modeChoose: 'Mode d\'entraînement',
    modeSimulDesc: 'Le discours source est lu pendant que vous interprétez et vous enregistrez en même temps.',
    modeConsecDesc: 'Écoutez la source et prenez des notes, puis enregistrez votre interprétation.',
    modeSightDesc: 'Lisez le texte défilant et traduisez à voix haute en vous enregistrant.',
    heroTrainIn: 'Entraînez-vous en :',
    heroTitlePre: "Maîtrisez l'interprétation avec des ",
    heroTitleEm: 'discours générés par IA',
    heroTitlePost: '',
    heroParagraph: "Une plateforme d'entraînement adaptative qui génère des discours simulant des situations de communication authentiques, élabore des glossaires multilingues et évalue vos performances en interprétation en arabe, en français et en anglais.",
    guestBtn: 'Continuer en invité',
    orSep: '— ou —',
    srcPanelTitle: 'Ajouter une source',
    srcPanelSubtitle: "Trouvez un document de l'ONU, une page web, ou déposez votre propre fichier pour ancrer le discours dans un contenu réel.",
    srcTabUN: 'Bibliothèque ONU',
    srcTabUpload: 'Déposer un fichier',
    srcUNHint: "Cherchez votre sujet dans la Bibliothèque numérique de l'ONU, téléchargez le PDF, puis déposez-le ci-dessous.",
    srcUNSearchBtn: 'Chercher dans la Bibliothèque ONU ↗',
    srcUNUploadAfter: 'Une fois le PDF téléchargé, déposez-le ici :',
    srcDropPdf: 'Déposez le PDF ici ou',
    srcDropFile: 'Déposez un fichier ici ou',
    srcBrowse: 'parcourir',
    srcFileTypes: 'PDF, Word (.docx) ou texte brut',
    audioFluencyTitle: "Fluidité audio de l'enregistrement",
    coverageScoreLabel: 'Score de couverture',
    statSpeechRate: 'Débit de parole',
    statLongPauses: 'Pauses longues',
    statSilenceRatio: 'Taux de silence',
    statWordConfidence: 'Clarté des mots',
    verdictGood: 'Bien',
    verdictAcceptable: 'Acceptable',
    verdictNeedsWork: 'À travailler',
    nextHintPre: '→ Allez dans',
    nextHintPost: 'pour écouter, réviser les questions et télécharger le glossaire.',
    transcriptReady: 'Transcription prête',
    chatGreeting: "Bonjour ! Je suis votre assistant ETIB. Posez-moi vos questions sur les techniques d'interprétation, la plateforme ou la terminologie.",
    chatPlaceholder: 'Posez une question…',
    chatSend: 'Envoyer',
    chatThinking: 'Réflexion…',
    statAvgPerSession: 'moy. / session',
    statTotal: 'total',
    statSessions: 'Sessions',
    outsideRange: '(hors de la plage demandée)',
    progressGuestNote: 'Le suivi de progression fonctionne avec un compte. Connectez-vous (ou créez un compte gratuit) pour que vos évaluations soient enregistrées et analysées.',
    pnTitle: 'Noms propres (personnes, organisations, lieux)',
    pnCorrect: 'Correct',
    pnDistorted: 'Déformé — probablement mal prononcé',
    pnMissing: 'Omis',
    llmFailedWarning: "L'analyse linguistique IA n'a pas pu se terminer — seules les détections automatiques sont affichées. Les scores peuvent être incomplets.",
    llmQuotaError: "Le service d'IA a atteint sa limite par minute — cela peut arriver avec de longs discours ou plusieurs requêtes d'affilée. La plateforme a déjà réessayé automatiquement ; attendez environ une minute puis réessayez. La limite partagée se réinitialise d'elle-même — rien à configurer.",
    keyInvalid: 'Le service d\'IA a rejeté la requête (la clé serveur de la plateforme est invalide ou expirée). Vous ne pouvez pas corriger cela — signalez-le à votre enseignant ou à l\'administrateur de la plateforme.',
    sameLangError: "La langue source et la langue d'interprétation doivent être différentes — l'interprétation se fait toujours entre deux langues.",
    simulSourceEnded: "Le discours source est terminé — achevez votre interprétation puis appuyez sur stop. L'enregistrement s'arrête automatiquement après une minute.",
    simulWithText: 'Afficher le texte source à l\'écran (SIMUL avec texte)',
    consecFlowHint: "Étape 1 : écoutez le discours et prenez des notes. Étape 2 : enregistrez votre interprétation à partir de vos notes ci-dessous.",
    simulUseButton: 'Utilisez « Lire la source + m\'enregistrer » ci-dessus pour démarrer.',
    sightUseButton: 'Utilisez le bouton d\'enregistrement à côté du défilement ci-dessus.',
    downloadRecording: 'Télécharger l\'enregistrement',
    downloadAudio: 'Télécharger l\'audio',
    llmDownError: "Le service d'évaluation IA n'a pas répondu — réessayez dans un instant.",
    addSource: 'Ajouter une source',
    uploadSourceMedia: 'Importer un discours à interpréter',
    sourceMediaEmpty: 'Impossible de transcrire le média importé — essayez un enregistrement plus clair.',
    micDenied: "Accès au microphone refusé — veuillez autoriser l'accès et réessayer.",
    simultaneousFailed: 'Impossible de démarrer le mode simultané — vérifiez l\'autorisation du microphone.',
    topErrorSaid: 'dit',
    topErrorExpected: 'Attendu',
    topErrorMissing: 'Manquant',
    uploadAudioBtn: 'Déposer un fichier audio',
    correctTashkeel: 'Tachkîl correct',
    iraabTitle: 'Désinences casuelles (i\'râb)',
    iraabAcousticNote: 'Détecté acoustiquement à partir de votre enregistrement — la transcription reste verbatim et les erreurs ne sont pas corrigées.',
    iraabDetected: 'Entendu',
    iraabCorrect: 'Correct',
    iraabIncorrect: 'Incorrect',
    iraabUncertain: 'Incertain',
    iraabAccuracy: 'Précision des désinences',
    iraabDiacritized: 'Hypothèse vocalisée',
  }
};

// ── Credits (professor request: ESIB + CINIA logos and developer names) ─────
const CREDITS = {
  developers: ['Joe Haddad', 'Kevin Khoury', 'Ali Ali', 'Chris Wehbe', 'Marc Khattar'],
  logos: [
    { src: '/usj-etib-logo.png',  alt: 'USJ — ETIB' },
    { src: '/usj-esib-logo.png',  alt: 'ESIB' },
    { src: '/usj-cinia-logo.png', alt: 'CINIA' },
  ],
};

function AppFooter({ labels }) {
  return (
    <footer style={{
      marginTop: '2.5rem', padding: '1.25rem 1.5rem', borderTop: '1px solid var(--border, #e3ded6)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1.25rem', flexWrap: 'wrap',
      fontSize: '0.8rem', color: 'var(--warm-gray, #877)',
    }}>
      {CREDITS.logos.map(logo => (
        <img key={logo.src} src={logo.src} alt={logo.alt} style={{ height: 34, objectFit: 'contain' }} />
      ))}
      <span>{labels.creditsLine}</span>
      <span>·</span>
      <span>{CREDITS.developers.join(' · ')}</span>
    </footer>
  );
}

const NAV_ITEMS = [
  { id: 'module-a', labelKey: 'navA' },
  { id: 'module-b', labelKey: 'navB' },
  { id: 'module-c', labelKey: 'navC' },
  { id: 'module-d', labelKey: 'navD' },
  { id: 'module-e', labelKey: 'navE' },
];

// MARC 041 language codes -> UI language codes / display labels
const UN_LANG_TO_UI = { ara: 'ar', fre: 'fr', eng: 'en' };
const UN_LANG_LABEL = {
  ara: 'AR', fre: 'FR', eng: 'EN',
  chi: 'ZH', rus: 'RU', spa: 'ES'
};

// Voice diversity expanded per professional-interpreter feedback (18/20 July):
// male voices + more regional accents in every language.
const VOICE_OPTIONS = {
  ar: [
    { label: 'Lebanese Arabic — Male (ar-LB)', labelAr: 'عربية لبنانية - ذكر', accent: 'LB' },
    { label: 'Lebanese Arabic — Female (ar-LB)', labelAr: 'عربية لبنانية - أنثى', accent: 'LB_f' },
    { label: 'Gulf Arabic — Female (ar-SA)', labelAr: 'عربية خليجية - أنثى', accent: 'SA' },
    { label: 'Gulf Arabic — Male (ar-SA)', labelAr: 'عربية خليجية - ذكر', accent: 'SA_m' },
    { label: 'Egyptian Arabic — Female (ar-EG)', labelAr: 'عربية مصرية - أنثى', accent: 'EG' },
    { label: 'Egyptian Arabic — Male (ar-EG)', labelAr: 'عربية مصرية - ذكر', accent: 'EG_m' },
    { label: 'Maghreb Arabic — Female (ar-MA)', labelAr: 'عربية مغاربية - أنثى', accent: 'MA' },
    { label: 'Maghreb Arabic — Male (ar-MA)', labelAr: 'عربية مغاربية - ذكر', accent: 'MA_m' },
  ],
  fr: [
    { label: 'French (Female)', labelAr: 'فرنسية - أنثى', accent: 'FR' },
    { label: 'French (Male)', labelAr: 'فرنسية - ذكر', accent: 'FR_m' },
    { label: 'Québécois (Female)', labelAr: 'كندية فرنسية - أنثى', accent: 'CA' },
    { label: 'Québécois (Male)', labelAr: 'كندية فرنسية - ذكر', accent: 'CA_m' },
  ],
  en: [
    { label: 'American (Female)', labelAr: 'أمريكية - أنثى', accent: 'US' },
    { label: 'British (Female)', labelAr: 'بريطانية - أنثى', accent: 'GB' },
    { label: 'British (Male)', labelAr: 'بريطانية - ذكر', accent: 'GB_m' },
    { label: 'Australian (Female)', labelAr: 'أسترالية - أنثى', accent: 'AU' },
    { label: 'Irish (Female)', labelAr: 'أيرلندية - أنثى', accent: 'IE' },
    { label: 'Irish (Male)', labelAr: 'أيرلندية - ذكر', accent: 'IE_m' },
    { label: 'Non-native accent (Female)', labelAr: 'لهجة غير أصلية - أنثى', accent: 'IN' },
    { label: 'Non-native accent (Male)', labelAr: 'لهجة غير أصلية - ذكر', accent: 'IN_m' },
  ]
};

// ── Glossary correction memory ───────────────────────────────────────────────
// Lina feedback (6 Aug 2026): "to what extent can the platform RETAIN the
// corrections the user makes, so glossary entries improve over time?"
// We persist every equivalent/definition the user corrects, keyed by the term,
// in localStorage. On each new generation these remembered corrections are
// re-applied to matching terms — so the same term is never wrong twice for a
// given user/browser. (Per-browser today; can be synced per-account later.)
const GLOSSARY_MEMORY_KEY = 'etib_glossary_memory_v1';
const GLOSSARY_MEMORY_FIELDS = ['arabic', 'french', 'english', 'definition'];

function normalizeGlossaryTerm(term) {
  return String(term || '').trim().toLowerCase();
}

function loadGlossaryMemory() {
  try {
    return JSON.parse(localStorage.getItem(GLOSSARY_MEMORY_KEY) || '{}') || {};
  } catch { return {}; }
}

// Save one corrected field for a term, together with the context it was
// corrected in: the language pair, the domain, and the fact that a human typed
// it. Re-editing the same term simply overwrites the stored value.
// Empty values are ignored — a cleared cell is not a correction and must never
// resurrect later.
function rememberGlossaryCorrection(term, field, value, context = {}) {
  const key = normalizeGlossaryTerm(term);
  if (!key || !GLOSSARY_MEMORY_FIELDS.includes(field)) return;
  const val = String(value ?? '').trim();
  if (!val) return;
  try {
    const memory = loadGlossaryMemory();
    memory[key] = {
      ...(memory[key] || {}),
      [field]: val,
      term: String(term).trim(),
      // Context of the correction. Kept per term (not per field) because the
      // student corrects a term while working on one speech, in one pair.
      source_language: context.source_language || memory[key]?.source_language || '',
      target_language: context.target_language || memory[key]?.target_language || '',
      domain:          context.domain          || memory[key]?.domain          || '',
      manually_edited: true,
      updated_at: new Date().toISOString(),
    };
    localStorage.setItem(GLOSSARY_MEMORY_KEY, JSON.stringify(memory));
  } catch { /* storage full / disabled — corrections just won't persist */ }
}

// Return how many remembered corrections exist (for the UI hint).
function glossaryMemorySize() {
  return Object.keys(loadGlossaryMemory()).length;
}

// Apply remembered corrections onto a fresh glossary list.
//
// A human correction is the HIGHEST authority: it overrides the model's own
// equivalent AND the one the institutional bases returned during server-side
// grounding. So a row patched from memory drops the `verified` marker coming
// from a base and is flagged `manually_edited` instead — the student must see
// whose value they are reading.
function applyGlossaryMemoryToList(glossary) {
  if (!Array.isArray(glossary) || !glossary.length) return glossary;
  const memory = loadGlossaryMemory();
  if (!Object.keys(memory).length) return glossary;
  return glossary.map(item => {
    const remembered = memory[normalizeGlossaryTerm(item?.term)];
    if (!remembered) return item;
    const patched = { ...item };
    let touched = false;
    for (const field of GLOSSARY_MEMORY_FIELDS) {
      if (remembered[field]) {
        patched[field] = remembered[field];
        // Kill the alias keys the display getters fall back through, or a stale
        // AI/base value could resurface over the student's own.
        for (const alias of (GLOSSARY_ALIASES[field] || [])) delete patched[alias];
        touched = true;
      }
    }
    if (touched) {
      patched.manually_edited = true;
      delete patched.verified;
      delete patched.verified_sources;
    }
    return patched;
  });
}

// Alias keys the LLM may use for the same field (kept in one place so the edit
// path and the memory path strip exactly the same set).
const GLOSSARY_ALIASES = {
  arabic:  ['Arabic', 'ar', 'AR', 'arabic_term', 'term_ar', 'arabic_translation', 'translation_ar', '\u0627\u0644\u0639\u0631\u0628\u064a\u0629', '\u0639\u0631\u0628\u064a'],
  french:  ['French', 'fr', 'FR', 'french_term', 'term_fr', 'french_translation', 'translation_fr', 'fran\u00e7ais', 'francais'],
  english: ['English', 'en', 'EN', 'english_term', 'term_en', 'english_translation', 'translation_en'],
  definition: [],
};

// Apply remembered corrections to a whole generated-script object's glossary.
function applyGlossaryMemory(script) {
  if (!script || !Array.isArray(script.glossary)) return script;
  return { ...script, glossary: applyGlossaryMemoryToList(script.glossary) };
}

// Merge server-side corrections (from the logged-in user's account) into the
// local memory map, so applyGlossaryMemory (which reads localStorage) picks
// them up. Called on login so a user's terms follow their account/device.
function mergeServerGlossaryCorrections(corrections) {
  if (!Array.isArray(corrections) || !corrections.length) return;
  try {
    const memory = loadGlossaryMemory();
    for (const c of corrections) {
      const key = normalizeGlossaryTerm(c?.term);
      if (!key) continue;
      const patch = { term: c.term, manually_edited: true };
      for (const f of GLOSSARY_MEMORY_FIELDS) if (c[f]) patch[f] = c[f];
      for (const f of ['source_language', 'target_language', 'domain', 'updated_at']) {
        if (c[f]) patch[f] = c[f];
      }
      memory[key] = { ...(memory[key] || {}), ...patch };
    }
    localStorage.setItem(GLOSSARY_MEMORY_KEY, JSON.stringify(memory));
  } catch { /* storage disabled — skip */ }
}

// Clear the local memory (on logout) so the next user on this browser does not
// inherit the previous account's remembered corrections.
function clearLocalGlossaryMemory() {
  try { localStorage.removeItem(GLOSSARY_MEMORY_KEY); } catch { /* ignore */ }
}

// ── SVG icons ────────────────────────────────────────────────────────────────
// Terminology is grounded SERVER-SIDE now (backend/services/terminology.py):
// each generated term is looked up in the UNBIS Thesaurus (UN, and the only
// machine-readable Arabic source), IATE (EU) and FranceTerme (official French),
// and an attested equivalent replaces the model's own. The old per-row 'check
// this term in 6 websites' links are gone — the check already happened.
// A row that a base confirmed carries `verified` + `verified_sources`.

function glossaryValue(item, keys) {
  for (const key of keys) {
    const value = item?.[key];
    if (value !== undefined && value !== null && String(value).trim()) {
      return String(value).trim();
    }
  }
  return '';
}

// Non-trimming getter for the EDIT inputs. The display getters call .trim(),
// which strips a trailing space the moment you type it — so the space never
// appears and the spacebar seems broken (professor feedback 20 July, Arabic
// glossary). During editing we read the exact key onChange writes to, raw.
function glossaryEditValue(item, canonicalKey, fallbackKeys) {
  if (item && typeof item[canonicalKey] === 'string') return item[canonicalKey];
  for (const key of fallbackKeys) {
    const v = item?.[key];
    if (v !== undefined && v !== null && typeof v !== 'object') return String(v);
  }
  return '';
}

function glossaryArabicValue(item) {
  // Student edits write the canonical 'arabic' key — they ALWAYS win.
  // (Tester bug Aug 2026: the IPCC hardcode below used to run first and
  // silently overwrite manual corrections.)
  if (typeof item?.arabic === 'string' && item.arabic.trim()) return item.arabic.trim();

  const combined = Object.values(item || {})
    .filter(value => value !== undefined && value !== null && typeof value !== 'object')
    .map(value => String(value).toLowerCase())
    .join(' ');
  if (combined.includes('giec') || combined.includes('ipcc')) {
    return 'الهيئة الحكومية الدولية المعنية بتغير المناخ';
  }

  const direct = glossaryValue(item, ['arabic', 'Arabic', 'ar', 'AR', 'arabic_term', 'term_ar', 'arabic_translation', 'translation_ar', 'العربية', 'عربي']);
  if (direct) return direct;

  for (const value of Object.values(item || {})) {
    if (value === undefined || value === null || typeof value === 'object') continue;
    const text = String(value).trim();
    if (/[\u0600-\u06FF]/.test(text)) return text;
  }
  return '';
}

const IconCopy = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg>
);
const IconThumbUp = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
  </svg>
);
const IconThumbDown = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
  </svg>
);
const IconRefresh = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
  </svg>
);
const IconTrash = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/>
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
    <line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/>
  </svg>
);

function TranscriptBubble({ labels, text, vocalizedText, isArabic, onRetranscribe, onDelete }) {
  const [copied, setCopied]       = useState(false);
  const [feedback, setFeedback]   = useState(null);
  const [showVocalized, setShowVocalized] = useState(true);

  const displayText = (isArabic && vocalizedText && showVocalized) ? vocalizedText : text;

  function copy() {
    navigator.clipboard.writeText(displayText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="message-bubble">
      {/* Tashkeel toggle — only for Arabic with vocalized version */}
      {isArabic && vocalizedText && (
        <div className="tashkeel-toggle">
          <button
            className={`tashkeel-btn ${showVocalized ? 'tashkeel-active' : ''}`}
            onClick={() => setShowVocalized(true)}
          >
            بِالتَّشْكِيل
          </button>
          <button
            className={`tashkeel-btn ${!showVocalized ? 'tashkeel-active' : ''}`}
            onClick={() => setShowVocalized(false)}
          >
            بدون تشكيل
          </button>
        </div>
      )}

      <div className={`bubble-text ${isArabic ? 'arabic' : ''}`}>{displayText}</div>

      <div className="bubble-actions">
        <button className={`bubble-btn ${copied ? 'bubble-btn-active' : ''}`} onClick={copy} title={labels.copyText}>
          <IconCopy />
        </button>
        <button className={`bubble-btn ${feedback === 'up' ? 'bubble-btn-active' : ''}`}
          onClick={() => setFeedback(f => f === 'up' ? null : 'up')} title={labels.bubbleGood}>
          <IconThumbUp />
        </button>
        <button className={`bubble-btn ${feedback === 'down' ? 'bubble-btn-active' : ''}`}
          onClick={() => setFeedback(f => f === 'down' ? null : 'down')} title={labels.bubblePoor}>
          <IconThumbDown />
        </button>
        {onRetranscribe && (
          <button className="bubble-btn" onClick={onRetranscribe} title={labels.redoTranscript}>
            <IconRefresh />
          </button>
        )}
        {onDelete && (
          <button className="bubble-btn bubble-btn-danger" onClick={onDelete} title={labels.deleteTranscript}>
            <IconTrash />
          </button>
        )}
      </div>
    </div>
  );
}

const initialSpeechForm = {
  topic: '',
  language: 'en',
  target_language: 'fr',
  domain: 'politics',
  word_count_range: 'medium',
  difficulty: 'intermediate',
  mode: 'consecutive',
  structure: 'well-organized',
  scenario: 'UN General Assembly',
  number_density: 'low',
  terminology_density: 'medium',
  include_hesitations: false,
  pressure_enabled: false,
  speed_pressure: 'normal',
  topic_shifts: 'none',
  context_noise: false,
  cognitive_load: 'medium'
};

// If the student pastes a full text instead of a short topic, treat the pasted
// text as the SOURCE DOCUMENT (grounding) rather than as a topic string.

const TOPIC_AS_SOURCE_THRESHOLD = 300;

// ── Module E — Progress & Adaptive Difficulty ─────────────────────────────────
function ScoreBadge({ score }) {
  const num = Number(score);
  const valid = Number.isFinite(num) && num > 0;
  const color = num >= 8 ? 'var(--sage)' : num >= 6.5 ? 'var(--primary)' : 'var(--sienna)';
  return (
    <span style={{ fontWeight: 700, color, fontFamily: 'Playfair Display, serif', fontSize: '1rem' }}>
      {valid ? num.toFixed(1) : '—'}
    </span>
  );
}

function MiniBar({ value, max = 10, color }) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  return (
    <div className="mini-bar-track">
      <div className="mini-bar-fill" style={{ width: `${pct}%`, background: color || 'var(--primary)' }} />
    </div>
  );
}

function TrendBadge({ trend, labels }) {
  if (!trend || trend === 'not_enough_data') return null;
  const map = {
    improving: { icon: '↑', color: 'var(--sage)',   text: labels.progressImproving },
    declining:  { icon: '↓', color: 'var(--sienna)', text: labels.progressDeclining },
    stable:     { icon: '→', color: 'var(--gold)',   text: labels.progressStable },
  };
  const d = map[trend];
  if (!d) return null;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.78rem',
      fontWeight: 700, color: d.color, background: d.color + '18', padding: '0.18rem 0.6rem',
      borderRadius: '999px', marginInlineStart: '0.5rem' }}>
      {d.icon} {d.text}
    </span>
  );
}

function SeverityDot({ severity }) {
  const c = severity === 'high' ? 'var(--sienna)' : severity === 'medium' ? 'var(--gold)' : 'var(--sage)';
  return <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: c, marginInlineEnd: '0.4rem', flexShrink: 0 }} />;
}

function ModuleProgress({ labels, refresh, onApplyParams }) {
  const [sessions, setSessions] = useState([]);
  const [adaptive, setAdaptive] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(null);
  const dateLocale = document.documentElement.lang === 'ar' ? 'ar-SA'
    : document.documentElement.lang === 'fr' ? 'fr-FR' : 'en-GB';

  useEffect(() => {
    setLoading(true);
    Promise.all([getSessionHistory(20), getAdaptiveParams()])
      .then(([hist, adap]) => {
        setSessions(hist.sessions || []);
        setAdaptive(adap);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [refresh]);

  // Plain language codes — flag emojis render as raw country codes (LB/GB/FR)
  // on Windows browsers, which reads as a mistake.
  const LANG_FLAG = { ar: 'AR', fr: 'FR', en: 'ENG' };

  // Sessions saved before a failed AI analysis carry 0/null scores — they are
  // not real results and must not drag averages down or show as "0.0" cards.
  const fmtScore = v => {
    const num = Number(v);
    return Number.isFinite(num) && num > 0 ? num.toFixed(1) : '—';
  };
  const avgOf = key => {
    const vals = sessions.map(s => Number(s?.[key])).filter(v => Number.isFinite(v) && v > 0);
    return vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2) : null;
  };
  const scoredSessions = sessions.filter(s => Number(s?.overall_score) > 0);
  const recentTen = scoredSessions.slice(0, 10).reverse();

  const n = scoredSessions.length;
  const avgOverall = avgOf('overall_score');
  const avgFluency = avgOf('fluency_score');
  const numberSessions = sessions.filter(s => s?.error_counts && Number.isFinite(Number(s.error_counts.number_errors)));
  const avgNumbers = numberSessions.length
    ? (numberSessions.reduce((s, x) => s + Number(x.error_counts.number_errors), 0) / numberSessions.length).toFixed(1)
    : null;

  const latest   = scoredSessions[0] || sessions[0] || null;
  const trend    = adaptive?.trend;
  const problems = adaptive?.problems_to_work_on || [];

  if (loading) {
    return <div className="card" style={{ textAlign: 'center', padding: '3rem' }}><div className="spinner" /><p style={{ marginTop: '1rem', color: 'var(--warm-gray)' }}>{labels.progressLoading}</p></div>;
  }

  return (
    <div className="module-b-layout">
      {/* ── Header stat row ── */}
      <div className="progress-stat-row">
        <div className="progress-stat-card">
          <div className="progress-stat-label">{labels.progressAvgOver}</div>
          <div className="progress-stat-value" style={{ color: avgOverall >= 7 ? 'var(--sage)' : avgOverall >= 5.5 ? 'var(--primary)' : 'var(--sienna)' }}>
            {avgOverall ?? '—'}
            {trend && trend !== 'not_enough_data' && <TrendBadge trend={trend} labels={labels} />}
          </div>
          <div className="progress-stat-sub">/ 10 &nbsp;{adaptive?.improvement_pct != null && trend !== 'not_enough_data' && (
            <span style={{ color: adaptive.improvement_pct > 0 ? 'var(--sage)' : 'var(--sienna)' }}>
              ({adaptive.improvement_pct > 0 ? '+' : ''}{adaptive.improvement_pct})
            </span>
          )}</div>
        </div>
        <div className="progress-stat-card">
          <div className="progress-stat-label">{labels.progressAvgFluency}</div>
          <div className="progress-stat-value" style={{ color: avgFluency >= 7 ? 'var(--sage)' : avgFluency >= 5.5 ? 'var(--primary)' : 'var(--sienna)' }}>{avgFluency ?? '—'}</div>
          <div className="progress-stat-sub">/ 10</div>
        </div>
        <div className="progress-stat-card">
          <div className="progress-stat-label">{labels.progressAvgNumbers}</div>
          <div className="progress-stat-value" style={{ color: avgNumbers <= 1 ? 'var(--sage)' : avgNumbers <= 3 ? 'var(--primary)' : 'var(--sienna)' }}>{avgNumbers ?? '—'}</div>
          <div className="progress-stat-sub">{labels.statAvgPerSession}</div>
        </div>
        <div className="progress-stat-card">
          <div className="progress-stat-label">{labels.statSessions}</div>
          <div className="progress-stat-value" style={{ color: 'var(--primary)' }}>{n}</div>
          <div className="progress-stat-sub">{labels.statTotal}</div>
        </div>
      </div>

      {n === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--warm-gray)' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>📊</div>
          <p>{labels.progressNoSessions}</p>
        </div>
      ) : (
        <>
          {/* ── Latest session card ── */}
          {latest && (
            <div className="card" style={{ borderInlineStart: '4px solid var(--primary)' }}>
              <h2 className="b-section-title">🎓 {labels.progressLatest}</h2>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
                <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                  <div className="progress-stat-card" style={{ minWidth: 90 }}>
                    <div className="progress-stat-label">{labels.progressOverall}</div>
                    <div className="progress-stat-value" style={{ color: Number(latest.overall_score) >= 7 ? 'var(--sage)' : Number(latest.overall_score) >= 5.5 ? 'var(--gold)' : 'var(--sienna)' }}>{fmtScore(latest.overall_score)}</div>
                  </div>
                  <div className="progress-stat-card" style={{ minWidth: 90 }}>
                    <div className="progress-stat-label">{labels.progressFluency}</div>
                    <div className="progress-stat-value" style={{ color: Number(latest.fluency_score) >= 7 ? 'var(--sage)' : 'var(--gold)' }}>{fmtScore(latest.fluency_score)}</div>
                  </div>
                  {/* Coverage only exists when meaning is in scope — hidden
                      otherwise rather than shown as a meaningless 0.0. */}
                  {latest.coverage_score != null && Number(latest.coverage_score) > 0 && (
                    <div className="progress-stat-card" style={{ minWidth: 90 }}>
                      <div className="progress-stat-label">{labels.progressCoverage}</div>
                      <div className="progress-stat-value" style={{ color: Number(latest.coverage_score) >= 7 ? 'var(--sage)' : 'var(--gold)' }}>{fmtScore(latest.coverage_score)}</div>
                    </div>
                  )}
                </div>
                <div style={{ flex: 1, minWidth: 200 }}>
                  {latest.summary && <p dir="auto" style={{ fontSize: '0.88rem', color: 'var(--warm-gray)', marginBottom: '0.5rem', fontStyle: 'italic' }}>"{latest.summary}"</p>}
                  {(latest.strengths || []).length > 0 && (
                    <div style={{ marginBottom: '0.4rem' }}>
                      <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--sage)' }}>{labels.progressStrengths}: </span>
                      <span dir="auto" style={{ fontSize: '0.82rem', color: 'var(--warm-gray)' }}>{latest.strengths.join(' · ')}</span>
                    </div>
                  )}
                  {(latest.recommendations || []).length > 0 && (
                    <div>
                      <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--gold)' }}>{labels.progressRecs}: </span>
                      <span dir="auto" style={{ fontSize: '0.82rem', color: 'var(--warm-gray)' }}>{latest.recommendations.slice(0, 2).join(' · ')}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ── Focus areas ── */}
          {problems.length > 0 && (
            <div className="card">
              <h2 className="b-section-title">🔍 {labels.progressProblems}</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {problems.map((p) => {
                  const sevLabel = labels[`severity${p.severity === 'high' ? 'High' : p.severity === 'medium' ? 'Medium' : 'Low'}`] || p.severity.toUpperCase();
                  const pLabel = labels[`problemLabel_${p.key}`] || p.label;
                  const pTip = labels[`problemTip_${p.key}`] || p.tip;
                  const detailTemplate = labels[`problemDetail_${p.key}`];
                  const detailMatch = (p.detail || '').match(/(\d+)\/(\d+)/);
                  const pDetail = detailTemplate && detailMatch
                    ? detailTemplate.replace('{count}', detailMatch[1]).replace('{total}', detailMatch[2])
                    : p.detail;
                  return (
                  <div key={p.key} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
                    background: 'var(--bg-secondary, #f8f7f5)', borderRadius: '8px', padding: '0.75rem 1rem' }}>
                    <span style={{ fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.05em', flexShrink: 0,
                      color: p.severity === 'high' ? 'var(--sienna)' : p.severity === 'medium' ? 'var(--gold)' : 'var(--sage)',
                      border: `1px solid currentColor`, borderRadius: '4px', padding: '0.15rem 0.4rem',
                      alignSelf: 'center' }}>
                      {sevLabel}
                    </span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary, #1a1a1a)' }}>{pLabel}</div>
                      <div dir="auto" style={{ fontSize: '0.8rem', color: 'var(--warm-gray)', marginTop: '0.1rem' }}>{pDetail}</div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--primary)', marginTop: '0.25rem' }}>💡 {pTip}</div>
                    </div>
                    <SeverityDot severity={p.severity} />
                  </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Score trend chart ── */}
          <div className="card">
            <h2 className="b-section-title">📈 {labels.progressTrend}</h2>
            <div className="trend-chart">
              {recentTen.map((s, i) => {
                const h = Math.round(((s.overall_score || 0) / 10) * 100);
                const color = s.overall_score >= 8 ? 'var(--sage)' : s.overall_score >= 6.5 ? 'var(--primary)' : 'var(--sienna)';
                return (
                  <div key={i} className="trend-bar-wrap">
                    <div className="trend-score-label" style={{ color }}>{fmtScore(s.overall_score)}</div>
                    <div className="trend-bar-outer">
                      <div className="trend-bar-inner" style={{ height: `${h}%`, background: color }} />
                    </div>
                    <div className="trend-lang-label">{LANG_FLAG[s.language] || '🌐'}</div>
                    <div className="trend-date-label">{new Date(s.created_at).toLocaleDateString(dateLocale, { day: '2-digit', month: '2-digit' })}</div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── Adaptive recommendations ── */}
          {adaptive?.recommended_params && (
            <div className="card">
              <h2 className="b-section-title">🎯 {labels.progressAdaptive}</h2>
              <p style={{ fontSize: '0.85rem', color: 'var(--warm-gray)', marginBottom: '1rem' }}>{labels.progressAdaptiveSubtitle}</p>
              <div className="adaptive-grid">
                {[
                  [labels.difficulty, adaptive.recommended_params.difficulty],
                  [labels.wordCount, adaptive.recommended_params.word_count + ' ' + labels.wordsUnit],
                  [labels.numbers, adaptive.recommended_params.number_density],
                  [labels.speedPressure, adaptive.recommended_params.speed_pressure],
                  [labels.structure, adaptive.recommended_params.structure],
                  [labels.topicShifts, adaptive.recommended_params.topic_shifts],
                ].map(([k, v]) => (
                  <div key={k} className="adaptive-param">
                    <div className="adaptive-param-label">{k}</div>
                    <div className="adaptive-param-value">{v}</div>
                  </div>
                ))}
              </div>
              {(adaptive.tips || []).length > 0 && (
                <ul className="adaptive-tips">
                  {adaptive.tips.map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              )}
              {onApplyParams && (
                <div style={{ marginTop: '1.25rem', textAlign: 'end' }}>
                  <button className="btn btn-primary" onClick={() => onApplyParams(adaptive.recommended_params)}>
                    {labels.progressApply}
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ── Session list ── */}
          <div className="card">
            <h2 className="b-section-title">📋 {labels.progressSessions}</h2>
            <div className="session-list">
              {scoredSessions.map((s, i) => (
                <div key={i} className={`session-row ${expanded === i ? 'session-row--open' : ''}`}
                  onClick={() => setExpanded(expanded === i ? null : i)}>
                  <div className="session-row-main">
                    <span className="session-lang">{LANG_FLAG[s.language] || (s.language || '').toUpperCase()}</span>
                    <span className="session-date">{new Date(s.created_at).toLocaleString(dateLocale, { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
                    <div className="session-scores">
                      <span title={labels.overallScore}><ScoreBadge score={s.overall_score} /></span>
                      <MiniBar value={s.overall_score} color={s.overall_score >= 7 ? 'var(--sage)' : s.overall_score >= 5.5 ? 'var(--primary)' : 'var(--sienna)'} />
                    </div>
                    <div className="session-error-chips">
                      {(s.error_counts?.number_errors > 0) && <span className="err-chip err-chip--num">🔢 {s.error_counts.number_errors}</span>}
                      {(s.error_counts?.repetitions > 0) && <span className="err-chip err-chip--rep">🔁 {s.error_counts.repetitions}</span>}
                      {(s.error_counts?.long_silences > 0) && <span className="err-chip err-chip--sil">🔇 {s.error_counts.long_silences}</span>}
                      {(s.error_counts?.translation_errors > 0) && <span className="err-chip err-chip--loss">🔤 {s.error_counts.translation_errors}</span>}
                    </div>
                    <span className="session-chevron">{expanded === i ? '▲' : '▼'}</span>
                  </div>
                  {expanded === i && (
                    <div className="session-detail">
                      <div className="session-detail-scores">
                        <div><span className="session-detail-label">{labels.progressOverall}</span><ScoreBadge score={s.overall_score} /></div>
                        <div><span className="session-detail-label">{labels.progressFluency}</span><ScoreBadge score={s.fluency_score} /></div>
                        {s.coverage_score != null && Number(s.coverage_score) > 0 && (
                          <div><span className="session-detail-label">{labels.progressCoverage}</span><ScoreBadge score={s.coverage_score} /></div>
                        )}
                      </div>
                      {s.summary && <p style={{ fontSize: '0.82rem', color: 'var(--warm-gray)', marginTop: '0.6rem', fontStyle: 'italic' }}>"{s.summary}"</p>}
                      {(s.strengths || []).length > 0 && (
                        <div style={{ marginTop: '0.5rem' }}>
                          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--sage)', textTransform: 'uppercase' }}>{labels.progressStrengths}</span>
                          <ul style={{ margin: '0.25rem 0 0 1rem', fontSize: '0.8rem', color: 'var(--warm-gray)' }}>
                            {s.strengths.map((r, j) => <li key={j}>{r}</li>)}
                          </ul>
                        </div>
                      )}
                      {(s.recommendations || []).length > 0 && (
                        <div style={{ marginTop: '0.5rem' }}>
                          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--gold)', textTransform: 'uppercase' }}>{labels.progressRecs}</span>
                          <ul style={{ margin: '0.25rem 0 0 1rem', fontSize: '0.8rem', color: 'var(--warm-gray)' }}>
                            {s.recommendations.map((r, j) => <li key={j}>{r}</li>)}
                          </ul>
                        </div>
                      )}
                      {(s.top_errors || []).length > 0 && (
                        <div style={{ marginTop: '0.5rem' }}>
                          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--sienna)', textTransform: 'uppercase' }}>{labels.progressTopErrors}</span>
                          <ul style={{ margin: '0.25rem 0 0 1rem', fontSize: '0.78rem', color: 'var(--warm-gray)' }}>
                            {s.top_errors.map((e, j) => (
                              <li key={j}>
                                {e.type === 'translation' && <><strong>{e.source}</strong> → {labels.topErrorSaid}: <em>{e.said}</em></>}
                                {e.type === 'number' && <>{labels.topErrorExpected}: <strong>{e.expected}</strong>, {labels.topErrorSaid}: <em>{e.said}</em></>}
                                {e.type === 'missing' && <>{labels.topErrorMissing}: <em>{e.detail}</em></>}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default function App() {
  const [uiLang, setUiLang] = useState('en');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [activePanel, setActivePanel] = useState('module-a');
  const [lastGeneratedScript, setLastGeneratedScript] = useState(null);
  const [sessionRestored, setSessionRestored] = useState(false);
  // Bumped on signup so the AI disclaimer modal opens for every new account.
  const [disclaimerTrigger, setDisclaimerTrigger] = useState(0);
  const L = UI[uiLang];

  // ── Session resume — ACCOUNTS ONLY (tester request, 5 Aug 2026) ───────────
  // A logged-in user's generated speech + materials + glossary edits survive a
  // reload / cold start and are restored on their NEXT login, keyed to their
  // account id. GUESTS are deliberately NOT persisted: nothing saved, nothing
  // restored, no banner. A live RECORDING never survives a reload (browser only).
  const SESSION_KEY_PREFIX = 'etib_session_snapshot_v1_';
  const isGuest = currentUser?.id === 'guest';
  const sessionKey = (isAuthenticated && !isGuest)
    ? SESSION_KEY_PREFIX + (currentUser?.id || 'anon')
    : null;

  // One-time cleanup of the legacy non-namespaced snapshot (may hold a guest's
  // speech from before this change).
  useEffect(() => { try { localStorage.removeItem('etib_session_snapshot_v1'); } catch { /* ignore */ } }, []);

  // Restore this account's last snapshot when they log in (never for guests).
  useEffect(() => {
    if (!sessionKey) return;
    try {
      const raw = localStorage.getItem(sessionKey);
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved?.script) { setLastGeneratedScript(saved); setSessionRestored(true); }
      }
    } catch { /* corrupt snapshot — ignore */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionKey]);

  // Persist on change — only for a real account (sessionKey null => guest or
  // logged out => never write, and logout keeps the snapshot for next login).
  useEffect(() => {
    if (!sessionKey) return;
    try {
      if (lastGeneratedScript?.script) {
        localStorage.setItem(sessionKey, JSON.stringify(lastGeneratedScript));
      } else {
        // Cleared to null while logged in (e.g. a failed/empty generation) —
        // wipe the snapshot so stale materials don't return on the next reload.
        localStorage.removeItem(sessionKey);
      }
    } catch { /* quota exceeded — not fatal */ }
  }, [lastGeneratedScript, sessionKey]);

  // Wake the free-tier backend immediately and keep it awake during the
  // session — a sleeping server (2-3 min cold start) caused testers'
  // transcription failures (18 July feedback).
  useEffect(() => {
    pingServer();
    const keepAlive = setInterval(pingServer, 10 * 60 * 1000);
    return () => clearInterval(keepAlive);
  }, []);

  useEffect(() => {
    const isAr = uiLang === 'ar';
    document.documentElement.lang = uiLang;
    document.documentElement.dir = isAr ? 'rtl' : 'ltr';
    document.body.classList.toggle('rtl', isAr);
  }, [uiLang]);

  // Pull this account's saved glossary corrections into local memory so they
  // are auto-applied to future generated glossaries (Lina 7 Aug: per-user
  // terminology memory tied to the profile, across sessions/devices).
  useEffect(() => {
    if (!currentUser || currentUser.id === 'guest') return;
    let cancelled = false;
    getGlossaryCorrections()
      .then(r => { if (!cancelled) mergeServerGlossaryCorrections(r?.corrections); })
      .catch(() => { /* offline / server waking — local memory still works */ });
    return () => { cancelled = true; };
  }, [currentUser]);

  function handleGuest() {
    saveAuthToken(null);
    setCurrentUserId('guest');
    setCurrentUser({ name: 'Guest', role: 'student', id: 'guest' });
    setIsAuthenticated(true);
  }

  async function handleLogin(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    try {
      const result = await loginUser({
        email: formData.get('email'),
        password: formData.get('password'),
        role: formData.get('role')
      });
      saveAuthToken(result.token);
      setCurrentUserId(result.user.id);
      setCurrentUser(result.user);
      setIsAuthenticated(true);
    } catch (err) {
      throw err; // let LoginScreen.handleSubmit surface the error
    }
  }

  async function handleSignup(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    try {
      const result = await signupUser({
        name: formData.get('name'),
        email: formData.get('email'),
        password: formData.get('password'),
        role: formData.get('role')
      });
      saveAuthToken(result.token);
      setCurrentUserId(result.user.id);
      setCurrentUser(result.user);
      setIsAuthenticated(true);
      // A new account must see the AI disclaimer in full, even on a browser
      // that already acknowledged it as a guest.
      setDisclaimerTrigger(n => n + 1);
    } catch (err) {
      throw err;
    }
  }

  async function handleLogout() {
    await logoutUser().catch(() => {});
    saveAuthToken(null);
    setCurrentUserId(null);
    setIsAuthenticated(false);
    setCurrentUser(null);
    setActivePanel('module-a');
    setLastGeneratedScript(null);
    setSessionRestored(false);
    // Clear this browser's glossary memory so the next user does not inherit
    // the previous account's remembered corrections (server stays the source
    // of truth and is re-fetched on the next login).
    clearLocalGlossaryMemory();
    // Keep the account's snapshot on disk so it can be restored on next login;
    // it is namespaced per user, so it never leaks to a guest or another account.
  }

  return (
    <div className="shell">
      <Header
        isAuthenticated={isAuthenticated}
        isGuest={currentUser?.id === 'guest'}
        activePanel={activePanel}
        labels={L}
        onPanelChange={setActivePanel}
        uiLang={uiLang}
        onLanguageChange={setUiLang}
      />
      <main>
        {isAuthenticated && !isGuest && sessionRestored && lastGeneratedScript?.script && (
          <div style={{ margin: '0.75rem auto', maxWidth: '1100px', padding: '0.6rem 1rem',
                        background: '#e8f2ec', border: '1px solid #b9d8c4', borderRadius: '8px',
                        display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap',
                        fontSize: '0.9rem' }}>
          <span style={{ flex: 1 }}>🔄 {L.sessionRestored}</span>
            <button className="btn-secondary btn-sm" onClick={() => {
              setLastGeneratedScript(null); setSessionRestored(false);
              try { if (sessionKey) localStorage.removeItem(sessionKey); } catch { /* ignore */ }
            }}>{L.sessionDiscard}</button>
            <button className="btn-secondary btn-sm" onClick={() => setSessionRestored(false)}>✓ {L.sessionKeep}</button>
          </div>
        )}
        {!isAuthenticated ? (
          <LoginScreen labels={L} onLogin={handleLogin} onSignup={handleSignup} onGuest={handleGuest} />
        ) : (
          <Workspace
            labels={L}
            activePanel={activePanel}
            onPanelChange={setActivePanel}
            onLogout={handleLogout}
            onGenerated={setLastGeneratedScript}
            lastGeneratedScript={lastGeneratedScript}
            currentUser={currentUser}
            isRtl={uiLang === 'ar'}
            disclaimerTrigger={disclaimerTrigger}
          />
        )}
      </main>
      <AppFooter labels={L} />
      {isAuthenticated && <ChatWidget labels={L} />}
    </div>
  );
}

function ChatWidget({ labels }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: labels.chatGreeting }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, open]);

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;
    const userMsg = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const apiMessages = [...messages, userMsg]
        .filter(m => m.role !== 'assistant' || messages.indexOf(m) > 0)
        .map(m => ({ role: m.role, content: m.content }));
      const { reply } = await sendChatMessage(apiMessages);
      setMessages(prev => [...prev, { role: 'assistant', content: reply }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ position: 'fixed', bottom: '1.5rem', insetInlineEnd: '1.5rem', zIndex: 1000 }}>
      {open && (
        <div style={{
          width: 340, height: 440, background: 'var(--surface, #fff)',
          border: '1px solid var(--border, #ddd)', borderRadius: 16,
          boxShadow: '0 8px 32px rgba(0,0,0,0.18)', display: 'flex',
          flexDirection: 'column', marginBottom: '0.75rem', overflow: 'hidden'
        }}>
          <div style={{
            background: 'var(--primary, #1a3a5c)', color: '#fff',
            padding: '0.85rem 1rem', fontWeight: 600, fontSize: '0.95rem',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center'
          }}>
            <span>ETIB Assistant</span>
            <button onClick={() => setOpen(false)} style={{
              background: 'none', border: 'none', color: '#fff',
              fontSize: '1.2rem', cursor: 'pointer', lineHeight: 1
            }}>×</button>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {messages.map((m, i) => (
              <div key={i} style={{
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                background: m.role === 'user' ? 'var(--primary, #1a3a5c)' : 'var(--surface-raised, #f0f4f8)',
                color: m.role === 'user' ? '#fff' : 'inherit',
                padding: '0.55rem 0.85rem', borderRadius: 12,
                maxWidth: '85%', fontSize: '0.875rem', lineHeight: 1.5,
                whiteSpace: 'pre-wrap'
              }}>
                {m.content}
              </div>
            ))}
            {loading && (
              <div style={{
                alignSelf: 'flex-start', background: 'var(--surface-raised, #f0f4f8)',
                padding: '0.55rem 0.85rem', borderRadius: 12, fontSize: '0.875rem',
                color: 'var(--warm-gray, #888)'
              }}>{labels.chatThinking}</div>
            )}
            <div ref={bottomRef} />
          </div>
          <div style={{ padding: '0.65rem', borderTop: '1px solid var(--border, #ddd)', display: 'flex', gap: '0.5rem' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder={labels.chatPlaceholder}
              style={{
                flex: 1, padding: '0.5rem 0.75rem', borderRadius: 8,
                border: '1px solid var(--border, #ddd)', fontSize: '0.875rem',
                outline: 'none', background: 'var(--bg, #fff)'
              }}
            />
            <button onClick={handleSend} disabled={loading || !input.trim()} style={{
              padding: '0.5rem 0.9rem', borderRadius: 8, border: 'none',
              background: 'var(--primary, #1a3a5c)', color: '#fff',
              cursor: 'pointer', fontWeight: 600, fontSize: '0.875rem',
              opacity: loading || !input.trim() ? 0.5 : 1
            }}>{labels.chatSend}</button>
          </div>
        </div>
      )}
      <button onClick={() => setOpen(o => !o)} style={{
        width: 52, height: 52, borderRadius: '50%',
        background: 'var(--primary, #1a3a5c)', color: '#fff',
        border: 'none', cursor: 'pointer', fontSize: '1.5rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
        display: 'flex', alignItems: 'center', justifyContent: 'center'
      }}>
        💬
      </button>
    </div>
  );
}

function Header({ isAuthenticated, isGuest, activePanel, labels, onPanelChange, uiLang, onLanguageChange }) {
  // Progress (module-e) is hidden from the UI for now (user request 11 Aug 2026):
  // the ModuleProgress component, its panel, and the backend history/adaptive
  // logic are all kept — to re-enable, restore the guest-only filter:
  //   NAV_ITEMS.filter(item => item.id !== 'module-e' || !isGuest)
  const navItems = NAV_ITEMS.filter(item => item.id !== 'module-e');
  return (
    <header className="fade-up">
      <div className="header-logos">
        <img src="/usj-etib-logo.png" style={{ height: '44px', objectFit: 'contain' }} alt="USJ - 150 ans - ETIB" />
        <div className="header-title">
          <div className="header-title-main">ETIB <span>Interpreter</span> Trainer</div>
          <div className="header-title-sub">AI Self-Training Platform · USJ Beirut</div>
        </div>
      </div>
      {isAuthenticated && (
        <nav aria-label="Training modules">
          {navItems.map(item => (
            <button
              key={item.id}
              type="button"
              className={`nav-btn ${activePanel === item.id ? 'active' : ''}`}
              onClick={() => onPanelChange(item.id)}
            >
              {labels[item.labelKey]}
            </button>
          ))}
        </nav>
      )}
      <label className="lang-picker">
        <span>{labels.uiLanguage}</span>
        <select value={uiLang} onChange={event => onLanguageChange(event.target.value)}>
          <option value="en">English</option>
          <option value="fr">Français</option>
          <option value="ar">العربية</option>
        </select>
      </label>
    </header>
  );
}

function LoginScreen({ labels, onLogin, onSignup, onGuest }) {
  const [mode, setMode] = useState('login');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const isSignup = mode === 'signup';

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setIsSubmitting(true);
    try {
      if (isSignup) {
        await onSignup(event);
      } else {
        await onLogin(event);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="hero fade-up delay-1">
      {/* Left: marketing copy */}
      <div className="hero-text">
        <div className="lang-tabs">
          <span className="lang-tabs-label">{labels.heroTrainIn}</span>
          <span className="lang-tab lang-tab-ar">عربي</span>
          <span className="lang-tab lang-tab-fr">Français</span>
          <span className="lang-tab lang-tab-en">English</span>
        </div>
        <h1>{labels.heroTitlePre}<em>{labels.heroTitleEm}</em>{labels.heroTitlePost}</h1>
        <p>{labels.heroParagraph}</p>
      </div>

      {/* Right: auth form */}
      <div className="login-panel">
        <h2>{isSignup ? labels.signupTitle : labels.loginTitle}</h2>
        <p className="sub">{isSignup ? labels.signupSubtitle : labels.loginSubtitle}</p>
        <form onSubmit={handleSubmit}>
          {isSignup && (
            <div className="field">
              <label htmlFor="login-name">{labels.name}</label>
              <input id="login-name" name="name" type="text" autoComplete="name" required placeholder={labels.namePlaceholder || 'Your full name'} />
            </div>
          )}
          <div className="field">
            <label htmlFor="login-email">{labels.email}</label>
            <input id="login-email" name="email" type="email" autoComplete="email" required placeholder="name@usj.edu.lb" />
          </div>
          <div className="field">
            <label htmlFor="login-password">{labels.password}</label>
            <div className="field-row">
              <input
                id="login-password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete="current-password"
                required
                minLength="4"
              />
              <button type="button" className="show-btn" onClick={() => setShowPassword(v => !v)}>
                {showPassword ? labels.hidePassword : labels.showPassword}
              </button>
            </div>
          </div>
          {error && <div className="error-msg">{labels.errorPrefix}: {error}</div>}
          <button type="submit" className="btn-primary login-submit" disabled={isSubmitting}>
            {isSubmitting ? (isSignup ? labels.signupLoading : labels.loginLoading) : (isSignup ? labels.signupSubmit : labels.loginSubmit)}
          </button>
          <p className="login-note">{labels.loginNote}</p>
          <div className="login-switch">
            <button type="button" onClick={() => { setMode(isSignup ? 'login' : 'signup'); setError(''); setShowPassword(false); }}>
              {isSignup ? labels.switchToLogin : labels.switchToSignup}
            </button>
          </div>
          <div style={{ textAlign: 'center', margin: '0.75rem 0 0' }}>
            <span style={{ color: '#aaa', fontSize: '0.8rem' }}>{labels.orSep}</span>
          </div>
          <button
            type="button"
            onClick={onGuest}
            style={{
              width: '100%', marginTop: '0.5rem', padding: '0.6rem',
              background: 'transparent', border: '1.5px solid #ccc',
              borderRadius: 8, cursor: 'pointer', fontSize: '0.9rem',
              color: '#555', fontWeight: 500,
            }}
          >
            {labels.guestBtn}
          </button>
        </form>
      </div>
    </section>
  );
}

// ── AI disclaimer (ETIB feedback, Lina — 21 Aug 2026) ────────────────────
// "Ne faudrait-il pas ajouter dans la plateforme un avertissement (AI disclaimer)
//  qui prévient les usagers des limites et de la nécessité de faire preuve de
//  vigilance lors de l'utilisation des sorties ?"
// The one-line warning is ALWAYS visible — it is not dismissible, because a
// disclaimer a student can hide forever stops being one. Only the detail is
// collapsible, and it stays open once opened for the session.
const AI_DISCLAIMER_ACK_KEY = 'etib_ai_disclaimer_ack_v1';

// ── AI disclaimer ─────────────────────────────────────────────
// Kept to ONE slim line in the workspace — the full text lives in a modal that
// opens itself once, on first use, and closes for good on "I understand".
// A permanent wall of text at the top of the page is not a usable disclaimer.
function AiDisclaimer({ labels, forceOpen = 0 }) {
  const [acked, setAcked] = useState(() => {
    try { return localStorage.getItem(AI_DISCLAIMER_ACK_KEY) === '1'; } catch { return false; }
  });
  const [open, setOpen] = useState(false);
  // Hiding the line is deliberately SESSION-ONLY (not persisted): the ETIB asked
  // for a warning that is present, so it comes back on the next visit.
  const [hidden, setHidden] = useState(false);

  // First visit: show the full text once, then never again unless asked for.
  useEffect(() => { if (!acked) setOpen(true); }, [acked]);

  // A brand-new account must read the disclaimer even if this browser already
  // acknowledged it as a guest — the counter re-fires on every signup.
  useEffect(() => {
    if (forceOpen > 0) { setHidden(false); setOpen(true); }
  }, [forceOpen]);

  function acknowledge() {
    try { localStorage.setItem(AI_DISCLAIMER_ACK_KEY, '1'); } catch { /* private mode */ }
    setAcked(true);
    setOpen(false);
  }

  return (
    <>
      {!hidden && (
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap',
        background: '#fdf6e7', border: '1px solid #f0e0be', borderRadius: 8,
        padding: '0.35rem 0.7rem', margin: '0 0 1rem 0', fontSize: '0.78rem',
        color: 'var(--warm-gray)', lineHeight: 1.4,
      }}>
        <span style={{ flex: 1, minWidth: 0 }}>⚠️ {labels.aiDisclaimerShort}</span>
        <button type="button" onClick={() => setOpen(true)} style={{
          background: 'none', border: 'none', padding: 0, cursor: 'pointer',
          color: 'var(--primary)', fontSize: '0.78rem', fontWeight: 600,
          textDecoration: 'underline', whiteSpace: 'nowrap',
        }}>{labels.aiDisclaimerMore}</button>
        <button type="button" onClick={() => setHidden(true)}
          title={labels.aiDisclaimerLess} aria-label={labels.aiDisclaimerLess} style={{
          background: 'none', border: 'none', padding: '0 0 0 0.2rem', cursor: 'pointer',
          color: 'var(--warm-gray)', fontSize: '1rem', lineHeight: 1,
          fontWeight: 700, whiteSpace: 'nowrap',
        }}>×</button>
      </div>
      )}

      {open && (
        <div onClick={() => acked && setOpen(false)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 1000,
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem',
        }}>
          <div onClick={e => e.stopPropagation()} style={{
            background: 'var(--cream, #fff)', borderRadius: 12, maxWidth: 680,
            width: '100%', maxHeight: '85vh', overflowY: 'auto', padding: '1.4rem 1.6rem',
            boxShadow: '0 12px 40px rgba(0,0,0,0.25)',
          }}>
            <h2 style={{ fontSize: '1.05rem', margin: '0 0 0.75rem' }}>{labels.aiDisclaimerTitle}</h2>
            <ul style={{ margin: 0, paddingInlineStart: '1.1rem', fontSize: '0.85rem', lineHeight: 1.65 }}>
              <li>{labels.aiDisclaimerGen}</li>
              <li>{labels.aiDisclaimerTts}</li>
              <li>{labels.aiDisclaimerStt}</li>
              <li>{labels.aiDisclaimerEval}</li>
              <li><strong>{labels.aiDisclaimerAdvice}</strong></li>
              <li>{labels.aboutEvalStorage}</li>
            </ul>
            <div style={{ display: 'flex', gap: '0.6rem', marginTop: '1.1rem', justifyContent: 'flex-end' }}>
              <button type="button" className="btn-primary" onClick={acknowledge}>
                {acked ? labels.aiDisclaimerLess : labels.aiDisclaimerAck}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function Workspace({ labels, activePanel, onPanelChange, onLogout, onGenerated, lastGeneratedScript, currentUser, isRtl, disclaimerTrigger = 0 }) {
  const [sharedAudioUrl, setSharedAudioUrl] = useState(null);
  const [lastTranscript, setLastTranscript] = useState(null);
  const [lastRecordingBlob, setLastRecordingBlob] = useState(null);
  const [progressRefresh, setProgressRefresh] = useState(0);
  const [adaptiveParams, setAdaptiveParams] = useState(null);

  return (
    <section>
      <div className="workspace-header fade-up">
        <span className="workspace-user">
          {currentUser && <><strong>{currentUser.id === 'guest' ? labels.guestName : currentUser.name}</strong> · {labels.student}</>}
        </span>
        <button type="button" className="sign-out-btn" onClick={onLogout}>{labels.signOut}</button>
      </div>

      {/* AI disclaimer (ETIB feedback 21 Aug 2026) — back at the top of the
          workspace, above the modules, where it is read before anything is used.
          The × hides the line for this session only; it returns on the next
          visit, and a new account always gets the full text as a modal. */}
      <AiDisclaimer labels={labels} forceOpen={disclaimerTrigger} />

      {/* Keep all panels mounted — state persists when switching tabs */}
      <div style={{ display: activePanel === 'module-a' ? 'block' : 'none' }}>
        <ModuleA labels={labels} onGenerated={onGenerated} lastGeneratedScript={lastGeneratedScript}
          isRtl={isRtl} adaptiveParams={adaptiveParams} onSourceAudio={setSharedAudioUrl} />
      </div>
      <div style={{ display: activePanel === 'module-b' ? 'block' : 'none' }}>
        <ModuleB labels={labels} lastGeneratedScript={lastGeneratedScript}
          onAudioGenerated={setSharedAudioUrl} onScriptUpdate={onGenerated} isRtl={isRtl} />
      </div>
      <div style={{ display: activePanel === 'module-c' ? 'block' : 'none' }}>
        <ModuleC labels={labels} referenceAudioUrl={sharedAudioUrl}
          sourceScript={lastGeneratedScript?.script || ''}
          targetLanguage={lastGeneratedScript?.target_language || ''}
          glossary={lastGeneratedScript?.glossary || []}
          onTranscriptComplete={setLastTranscript}
          onRecordingComplete={setLastRecordingBlob} />
      </div>
      <div style={{ display: activePanel === 'module-d' ? 'block' : 'none' }}>
        <ModuleD labels={labels} lastTranscript={lastTranscript}
          lastGeneratedScript={lastGeneratedScript}
          lastRecordingBlob={lastRecordingBlob}
          onEvaluationSaved={() => setProgressRefresh(n => n + 1)} />
      </div>
      <div style={{ display: activePanel === 'module-e' ? 'block' : 'none' }}>
        {currentUser?.id === 'guest' ? (
          <div className="card coming-soon">
            <p className="eyebrow">{labels.navE}</p>
            <p>{labels.progressGuestNote}</p>
          </div>
        ) : (
        <ModuleProgress labels={labels} refresh={progressRefresh}
          onApplyParams={(params) => {
            // Adaptive difficulty (cahier D12): apply the recommended
            // parameters to the generation form, then jump to Module A.
            setAdaptiveParams({ ...params, _appliedAt: Date.now() });
            onPanelChange('module-a');
          }} />
        )}
      </div>

    </section>
  );
}

// ── Sources Panel ─────────────────────────────────────────────────────────────

function SourcesPanel({ labels, language, domain, initialQuery, onSelectLibrary, onSelectFile, onClose }) {
  const [tab, setTab]   = useState('library');  // 'library' | 'upload' | 'webpage'
  const [query, setQuery] = useState(initialQuery || '');
  const [pageUrl, setPageUrl] = useState('');
  const [urlStatus, setUrlStatus] = useState('idle');
  const [urlError, setUrlError] = useState('');
  const [urlSuccessTitle, setUrlSuccessTitle] = useState('');
  const [fileError, setFileError] = useState('');

  // Only these document types are supported by the backend
  // (utils/document_grounding.SUPPORTED_DOCUMENT_EXTENSIONS).
  const ALLOWED_DOC_EXT = ['.pdf', '.docx', '.txt'];

  function handleFileChosen(file) {
    if (!file) return;
    const name = (file.name || '').toLowerCase();
    if (!ALLOWED_DOC_EXT.some(ext => name.endsWith(ext))) {
      // Client-side type check: the accept="" attribute is only a picker hint
      // and does not block drag-drop or every browser, so validate explicitly.
      setFileError(labels.fileTypeError || 'Unsupported file type. Upload a PDF, Word (.docx), or plain-text file.');
      return;
    }
    setFileError('');
    onSelectFile(file);
    onClose();
  }

  async function handleFetchUrl() {
    if (!pageUrl.trim()) return;
    setUrlStatus('loading'); setUrlError(''); setUrlSuccessTitle('');
    try {
      const data = await fetchWebPage(pageUrl.trim());
      const title = data.title || pageUrl;
      onSelectLibrary({ text: data.text, title, un_id: '', web_url: data.url });
      // Explicit success feedback: the tester reported the modal closing with no
      // visible confirmation. Keep it open and confirm what was attached.
      setUrlSuccessTitle(title);
      setUrlStatus('success');
      setPageUrl('');
    } catch (err) {
      setUrlError(err.message);
      setUrlStatus('error');
    }
  }

  const unSearchUrl = `https://digitallibrary.un.org/search?p=${encodeURIComponent(query || 'United Nations')}&ln=en&sf=date&so=d&of=hb&fct__1=Documents+and+Publications`;

  return (
    <div className="library-overlay">
      <div className="library-panel">
        <div className="library-header">
          <h2 className="library-title">{labels?.srcPanelTitle || 'Add Source'}</h2>
          <button className="library-close" onClick={onClose} aria-label={labels.closeLabel}>✕</button>
        </div>
        <p className="library-subtitle">{labels?.srcPanelSubtitle}</p>

        <div className="library-tabs">
          <button className={`lib-tab ${tab === 'library' ? 'lib-tab-active' : ''}`} onClick={() => setTab('library')}>{labels?.srcTabUN || 'UN Library'}</button>
          <button className={`lib-tab ${tab === 'webpage' ? 'lib-tab-active' : ''}`} onClick={() => setTab('webpage')}>🔗 {labels?.webPageTab || 'Web page'}</button>
          <button className={`lib-tab ${tab === 'upload' ? 'lib-tab-active' : ''}`} onClick={() => setTab('upload')}>{labels?.srcTabUpload || 'Upload file'}</button>
        </div>

        {tab === 'webpage' && (
          <div style={{ padding: '1.2rem 0' }}>
            <p style={{ fontSize: '0.875rem', color: 'var(--warm-gray)', marginBottom: '1rem', lineHeight: 1.6 }}>
              {labels?.webPageHint || 'The readable text of the page will be extracted and used as the source for the speech.'}
            </p>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                className="library-search-input"
                type="url"
                value={pageUrl}
                onChange={e => { setPageUrl(e.target.value); setUrlError(''); setUrlStatus('idle'); }}
                placeholder={labels?.webPageUrlPlaceholder || 'https://…'}
                onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); handleFetchUrl(); } }}
                disabled={urlStatus === 'loading'}
              />
              <button
                type="button"
                className="btn-primary"
                style={{ whiteSpace: 'nowrap' }}
                onClick={handleFetchUrl}
                disabled={urlStatus === 'loading' || !pageUrl.trim()}
              >
                {urlStatus === 'loading' ? (labels?.webPageFetching || 'Fetching…') : (labels?.webPageFetch || 'Fetch page')}
              </button>
            </div>
            {urlError && <div className="error-msg" style={{ marginTop: '0.75rem' }}>{urlError}</div>}
            {urlStatus === 'success' && (
              <div className="info-tip" style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                <span style={{ flex: 1 }}>✅ {labels.webPageAttached}: <strong>{urlSuccessTitle}</strong></span>
                <button type="button" className="btn-primary" style={{ whiteSpace: 'nowrap' }} onClick={onClose}>
                  {labels.webPageDone || 'Done'}
                </button>
              </div>
            )}
          </div>
        )}

        {tab === 'library' && (
          <div style={{ padding: '1.2rem 0' }}>
            <p style={{ fontSize: '0.875rem', color: 'var(--warm-gray)', marginBottom: '1rem', lineHeight: 1.6 }}>
              {labels?.srcUNHint}
            </p>

            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.2rem' }}>
              <input
                className="library-search-input"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder={labels.librarySearchPlaceholder || 'e.g. climate change, AI governance…'}
                onKeyDown={e => { if (e.key === 'Enter') window.open(unSearchUrl, '_blank'); }}
              />
              <a
                href={unSearchUrl}
                target="_blank"
                rel="noreferrer"
                className="btn-primary"
                style={{ whiteSpace: 'nowrap', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}
              >
                {labels?.srcUNSearchBtn || 'Search UN Library ↗'}
              </a>
            </div>

            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.1rem' }}>
              <p style={{ fontSize: '0.82rem', color: 'var(--warm-gray)', marginBottom: '0.75rem' }}>
                {labels?.srcUNUploadAfter}
              </p>
              <label
                className="upload-zone"
                onDragOver={e => { e.preventDefault(); e.stopPropagation(); }}
                onDrop={e => { e.preventDefault(); e.stopPropagation(); handleFileChosen(e.dataTransfer.files?.[0] || null); }}
              >
                <input
                  type="file"
                  accept=".pdf,.txt,.docx"
                  onChange={e => handleFileChosen(e.target.files?.[0] || null)}
                  style={{ display: 'none' }}
                />
                <svg className="upload-zone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <div className="upload-zone-label">{labels?.srcDropPdf} <span className="upload-zone-browse">{labels?.srcBrowse}</span></div>
                <div className="upload-zone-hint">{labels?.srcFileTypes}</div>
              </label>
              {fileError && <div className="error-msg" style={{ marginTop: '0.75rem' }}>{fileError}</div>}
            </div>
          </div>
        )}

        {tab === 'upload' && (
          <div className="library-results" style={{ padding: '0.9rem 0 0' }}>
            <label
              className="upload-zone"
              onDragOver={e => { e.preventDefault(); e.stopPropagation(); }}
              onDrop={e => { e.preventDefault(); e.stopPropagation(); handleFileChosen(e.dataTransfer.files?.[0] || null); }}
            >
              <input
                type="file"
                accept=".txt,.docx,.pdf"
                onChange={e => handleFileChosen(e.target.files?.[0] || null)}
                style={{ display: 'none' }}
              />
              <svg className="upload-zone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <div className="upload-zone-label">{labels?.srcDropFile} <span className="upload-zone-browse">{labels?.srcBrowse}</span></div>
              <div className="upload-zone-hint">{labels?.srcFileTypes}</div>
            </label>
            {fileError && <div className="error-msg" style={{ marginTop: '0.75rem' }}>{fileError}</div>}
          </div>
        )}
      </div>
    </div>
  );
}

function ModuleA({ labels, onGenerated, lastGeneratedScript, isRtl, adaptiveParams, onSourceAudio }) {
  const [form, setForm] = useState(initialSpeechForm);
  const sourceMediaRef = useRef(null);
  const [sourceMediaStatus, setSourceMediaStatus] = useState('idle');

  // Adaptive difficulty (cahier D12): merge the recommended parameters from
  // the Progress page into the generation form whenever the student clicks
  // "Apply these settings".
  useEffect(() => {
    if (!adaptiveParams) return;
    const wc = Number(adaptiveParams.word_count || 0);
    const word_count_range = wc <= 180 ? 'short' : wc <= 320 ? 'medium' : 'long';
    setForm(current => ({
      ...current,
      difficulty: adaptiveParams.difficulty || current.difficulty,
      word_count_range: wc ? word_count_range : current.word_count_range,
      number_density: adaptiveParams.number_density || current.number_density,
      speed_pressure: adaptiveParams.speed_pressure || current.speed_pressure,
      structure: adaptiveParams.structure || current.structure,
      topic_shifts: adaptiveParams.topic_shifts || current.topic_shifts,
    }));
  }, [adaptiveParams]);
  const [documentFiles, setDocumentFiles] = useState([]);   // File[]
  const [librarySources, setLibrarySources] = useState([]); // {text, title, un_id}[]
const [showAdvanced, setShowAdvanced] = useState(true);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [showLibrary, setShowLibrary] = useState(false);
  const isLoading = status === 'loading';

  // "Use my own text" (ETIB feedback 21 Aug 2026, Lina: "est-il possible
  // d'ajouter une option qui permet aux étudiants d'ajouter un texte tel quel
  // sans avoir à passer par l'IA, c'est-à-dire pour générer l'audio seulement ?").
  // The text is taken VERBATIM — no generation, no rewriting, no proofreading
  // pass — so none of the LLM's text errors, biases or hallucinations can enter
  // the exercise. Only the TTS step runs afterwards.
  const [ownTextMode, setOwnTextMode] = useState(false);
  const [ownText, setOwnText] = useState('');
  const [editStatus, setEditStatus] = useState('idle');

  function handleUseOwnText() {
    const text = ownText.trim();
    if (!text) { setError(labels.ownTextEmpty); setStatus('error'); return; }
    if (form.language === form.target_language) { setError(labels.sameLangError); setStatus('error'); return; }
    const words = text.split(/\s+/).filter(Boolean);
    const generated = {
      script:          text,
      language:        form.language,
      target_language: form.target_language,
      domain:          form.domain,
      topic:           words.slice(0, 8).join(' '),
      word_count:      words.length,
      summary: '', mcqs: [], glossary: [],
      ai_generated: false,
      mode: 'own_text',
    };
    setError(''); setResult(generated); onGenerated(generated); setStatus('success');
  }

  function updateField(event) {
    const { name, value, type, checked } = event.target;
    // NOTE: we deliberately do NOT auto-clear attached sources when the topic
    // or domain changes. Doing so silently dropped the user's chosen source
    // (Lina feedback 6 Aug: "the generated text was based on a UN document I
    // did not choose") — because once the source was wiped, generation fell
    // back to the auto UN-library lookup. Sources now persist until the user
    // removes them explicitly via the chip's × button.
    setForm(current => ({ ...current, [name]: type === 'checkbox' ? checked : value }));
  }

  const topicIsLongText = form.topic.trim().length > TOPIC_AS_SOURCE_THRESHOLD;

  async function handleSubmit(event) {
    event.preventDefault();
    // Personal Groq key is mandatory (16 July feedback) — no shared-quota use.
    // Interpretation is cross-language: source and target must differ.
    if (form.language === form.target_language) {
      setError(labels.sameLangError); setStatus('error'); return;
    }
    // Reset any previous/restored session up front so a new generation can
    // never leave stale summary/MCQ from an earlier speech on screen.
    setStatus('loading'); setError(''); setResult(null); onGenerated(null);
    try {
      let params = form;
      if (topicIsLongText) {
        // Pasted full text → use it as the grounding source document and
        // derive a short topic from its first words.
        const pasted = form.topic.trim();
        params = {
          ...form,
          topic: pasted.split(/\s+/).slice(0, 12).join(' '),
          source_text: pasted,
          source_title: 'Pasted text',
        };
      }
      const data = await generateSpeech(params);
      if (!(data?.script || '').trim()) {
        setResult(null); onGenerated(null);
        setError(labels.generationEmpty || 'No speech text was generated — please try again.');
        setStatus('error'); return;
      }
      const remembered = applyGlossaryMemory(data);
      setResult(remembered); onGenerated(remembered); setStatus('success');
    } catch (err) { setResult(null); onGenerated(null); setError(friendlyLlmError(err.message, labels)); setStatus('error'); }
  }

  // Upload a real speech (audio or video) to interpret from (Mariam feedback):
  // transcribe it → becomes the source script, and its audio plays in Module C.
  async function handleSourceMediaUpload(file) {
    if (!file) return;
    setSourceMediaStatus('loading'); setError(''); setResult(null); onGenerated(null);
    try {
      // Reuse the ASR endpoint (it now normalizes audio so the FULL media is
      // transcribed, video included — the container is decoded server-side).
      const data = await transcribeAudio(file, form.language, '');
      const script = (data.full_text || '').trim();
      if (!script) throw new Error(labels.sourceMediaEmpty || 'Could not transcribe the uploaded media.');
      const audioUrl = URL.createObjectURL(file);
      onSourceAudio?.(audioUrl);   // play the real source in Module C

      // Generate the full materials from the transcript so the uploaded speech
      // behaves exactly like a generated one — audio, MCQ, glossary all work.
      // Not silent: if this fails the student sees why (usually a sleeping
      // server or an exhausted key) and can retry.
      setSourceMediaStatus('materials');
      let materials = { summary: '', mcqs: [], glossary: [] };
      let materialsError = '';
      try {
        materials = await generateMaterialsFromScript({ script, language: form.language, domain: form.domain });
      } catch (mErr) {
        materialsError = mErr.message;
      }

      const generated = {
        script,
        language: form.language,
        target_language: form.target_language,
        domain: form.domain,
        summary:  materials.summary || '',
        mcqs:     materials.mcqs || [],
        glossary: materials.glossary || [],
        source_upload: true,
        mode: 'uploaded_source',
      };
      const rememberedGen = applyGlossaryMemory(generated);
      setResult(rememberedGen); onGenerated(rememberedGen);
      if (materialsError) {
        // Surface a clear reason (invalid key, quota) instead of raw JSON.
        setError(`${labels.errorPrefix}: ${friendlyLlmError(materialsError, labels)}`);
        setSourceMediaStatus('error');
      } else if (!(materials.mcqs || []).length) {
        setError(labels.materialsFailed || 'The speech was transcribed, but the materials could not be generated — please retry.');
        setSourceMediaStatus('error');
      } else {
        setSourceMediaStatus('success');
      }
    } catch (err) {
      setError(friendlyLlmError(err.message, labels)); setSourceMediaStatus('error');
    }
  }

  // Saving an edited speech re-derives the pedagogical materials from the NEW
  // text: a changed figure, a reworded passage or a half-rewritten speech would
  // otherwise leave the summary, MCQs and glossary describing the old version.
  // The glossary the student had already corrected is preserved by term.
  async function handleSaveScriptEdit(newText) {
    const text = String(newText || '').trim();
    const current = result || lastGeneratedScript;
    if (!text || !current || text === current.script) return;

    const base = {
      ...current,
      script: text,
      script_original: current.script_original ?? current.script,
      script_edited: true,
      word_count: text.split(/\s+/).filter(Boolean).length,
    };
    setResult(base); onGenerated(base);

    // An edited text no longer matches its materials — rebuild them.
    setEditStatus('loading');
    try {
      const mats = await generateMaterialsFromScript({
        script: text,
        language: base.language,
        domain: base.domain || 'general',
      });
      const previous = current.glossary || [];
      const keepEdits = new Map(
        previous.filter(g => g && g.term).map(g => [String(g.term).trim().toLowerCase(), g])
      );
      const merged = (mats.glossary || []).map(row => {
        const kept = keepEdits.get(String(row.term || '').trim().toLowerCase());
        return kept ? { ...row, ...kept } : row;
      });
      const updated = applyGlossaryMemory({
        ...base,
        summary:  mats.summary || base.summary || '',
        mcqs:     (mats.mcqs || []).length ? mats.mcqs : base.mcqs,
        glossary: merged.length ? merged : base.glossary,
      });
      setResult(updated); onGenerated(updated);
      setEditStatus('idle');
    } catch (err) {
      // The edited text is already saved; only the refresh failed.
      setEditStatus('error');
      setError(`${labels.errorPrefix}: ${friendlyLlmError(err.message, labels)}`);
    }
  }

  function buildAllSourceFiles() {
    const libFiles = librarySources.map(src => {
      const blob = new Blob([src.text], { type: 'text/plain' });
      return new File([blob], `${src.un_id || 'source'}.txt`, { type: 'text/plain' });
    });
    return [...documentFiles, ...libFiles];
  }

  const hasSources = documentFiles.length > 0 || librarySources.length > 0;

  async function handleDocumentGenerate() {
    if (form.language === form.target_language) {
      setError(labels.sameLangError); setStatus('error'); return;
    }
    setStatus('loading'); setError(''); setResult(null); onGenerated(null);
    try {
      const allFiles = buildAllSourceFiles();
      if (!allFiles.length) throw new Error(labels.addSourceFirst || 'Add at least one source document first.');
      const topicFallback = form.topic || librarySources[0]?.title || '';
      const data = await generateSpeechFromDocument(allFiles, { ...form, topic: topicFallback });
      if (!(data?.script || '').trim()) {
        setResult(null); onGenerated(null);
        setError(labels.generationEmpty || 'No speech text was generated — please try again.');
        setStatus('error'); return;
      }
      const remembered = applyGlossaryMemory(data);
      setResult(remembered); onGenerated(remembered); setStatus('success');
    } catch (err) { setResult(null); onGenerated(null); setError(friendlyLlmError(err.message, labels)); setStatus('error'); }
  }

  const LANG_LABEL = { ar: 'AR', fr: 'FR', en: 'EN' };

  return (
    <div className="card module-a-card">
      <form onSubmit={handleSubmit}>

        {/* ── Main topic bar — textarea so full texts can be pasted ── */}
        <div className="topic-bar">
          <textarea
            className="topic-input"
            name="topic"
            value={form.topic}
            minLength="3"
            maxLength="12000"
            required
            rows={topicIsLongText ? 6 : 1}
            placeholder={labels.topicPlaceholder || 'Enter a topic or paste text to generate a speech…'}
            onChange={updateField}
            disabled={isLoading}
            style={{ resize: 'vertical', overflow: 'auto', minHeight: '2.6rem' }}
          />
        </div>
        {topicIsLongText && (
          <div className="info-tip" style={{ marginTop: '0.4rem' }}>📄 {labels.longTextAsSource}</div>
        )}

        {/* ── Attached source chips ── */}
        {documentFiles.map((f, i) => (
          <div key={i} className="file-chip">
            <span>📄 {f.name}</span>
            <button type="button" className="file-chip-remove" onClick={() => {
              setDocumentFiles(prev => prev.filter((_, idx) => idx !== i));
              setResult(null); setError('');
            }}>×</button>
          </div>
        ))}
        {librarySources.map((src, i) => (
          <div key={i} className="file-chip file-chip-un">
            <span>📚 {src.title.slice(0, 60)}{src.title.length > 60 ? '…' : ''}</span>
            <button type="button" className="file-chip-remove" onClick={() => {
              setLibrarySources(prev => prev.filter((_, idx) => idx !== i));
              setResult(null); setError('');
            }} title={labels.removeSource || 'Remove source'}>×</button>
          </div>
        ))}

        {/* ── Language + quick settings row ── */}
        <div className="quick-settings-row">
          <div className="lang-pair">
            <select name="language" value={form.language} onChange={updateField} className="lang-select" title={labels.language}>
              <option value="ar">AR</option>
              <option value="fr">FR</option>
              <option value="en">EN</option>
            </select>
            <span className="lang-arrow">{isRtl ? '←' : '→'}</span>
            <select name="target_language" value={form.target_language} onChange={updateField} className="lang-select" title={labels.targetLanguage}>
              <option value="ar">AR</option>
              <option value="fr">FR</option>
              <option value="en">EN</option>
            </select>
          </div>

          <select name="difficulty" value={form.difficulty} onChange={updateField} className="quick-select">
            <option value="beginner">{labels.diffBeginner || 'Beginner'}</option>
            <option value="intermediate">{labels.diffIntermediate || 'Intermediate'}</option>
            <option value="advanced">{labels.diffAdvanced || 'Advanced'}</option>
          </select>

          <button
            type="button"
            className={`advanced-toggle ${showAdvanced ? 'advanced-toggle-active' : ''}`}
            onClick={() => setShowAdvanced(v => !v)}
          >
            {showAdvanced ? (labels.lessSettings || '⚙ Less') : (labels.moreSettings || '⚙ More')}
          </button>
        </div>

        {/* Interpretation is cross-language — warn on an identical pair. */}
        {form.language === form.target_language && (
          <div className="error-msg" style={{ marginTop: '0.5rem' }}>⚠️ {labels.sameLangError}</div>
        )}

        {/* ── Advanced options (collapsible) ── */}
        {showAdvanced && (
          <div className="advanced-panel">
            <div className="form-grid">
              <SelectField label={labels.wordCount} id="f-words" name="word_count_range" value={form.word_count_range} onChange={updateField}>
                <option value="short">{labels.wordRangeShortFull || 'Short — 120–180 words (≈1–1.5 min)'}</option>
                <option value="medium">{labels.wordRangeMediumFull || 'Medium — 220–320 words (≈2–2.5 min)'}</option>
                <option value="long">{labels.wordRangeLongFull || 'Long — 400–550 words (≈3.5–4.5 min)'}</option>
                {/* Lengths above "Long" removed from the dropdown (user request).
                    ARCHIVED — the backend WORD_COUNT_RANGES presets and the label
                    strings are kept, so restore by pasting the <option> back:
                      <option value="extended">{labels.wordRangeExtendedFull}</option>
                      <option value="very_long">{labels.wordRangeVeryLongFull}</option>
                      <option value="extra_long">{labels.wordRangeExtraLongFull}</option>
                      <option value="marathon">{labels.wordRangeMarathonFull}</option> */}
              </SelectField>
              {/* Domains removed from the dropdown (10 Aug 2026, user request).
                  ARCHIVED — backend DOMAIN_QUERIES entries and the labels are kept,
                  so restore by pasting the <option> back:
                    <option value="migration">{labels.domMigration || 'Migration & Refugees'}</option>
                    <option value="women">{labels.domWomen || 'Women & Gender'}</option>
                    <option value="food">{labels.domFood || 'Food & Hunger'}</option>
                    <option value="legal">{labels.domLegal || 'Legal'}</option>
                    <option value="medical">{labels.domMedical || 'Medical'}</option>
                    <option value="diplomacy">{labels.domDiplomacy || 'Diplomacy'}</option>
                    <option value="human rights">{labels.domHumanRights || 'Human Rights'}</option> */}
              <SelectField label={labels.domain} id="f-domain" name="domain" value={form.domain} onChange={updateField}>
                <option value="politics">{labels.domPolitics || 'Politics'}</option>
                <option value="economics">{labels.domEconomics || 'Economics'}</option>
                <option value="climate">{labels.domClimate || 'Climate & Environment'}</option>
                <option value="health">{labels.domHealth || 'Health'}</option>
                <option value="education">{labels.domEducation || 'Education'}</option>
                <option value="technology">{labels.domTechnology || 'Technology & AI'}</option>
                <option value="disarmament">{labels.domDisarmament || 'Disarmament'}</option>
              </SelectField>
              {/* Scenario list trimmed from 19 to 7 curated, verified styles
                  (10 Aug 2026 — most were redundant institutional podium variants
                  that differ only in vocabulary, already controlled by Domain/Topic).
                  Backend SCENARIO_STYLES/SCENARIO_FORMATS keep ALL definitions, so any
                  archived scenario is restored by pasting its <option> back here.

                  ARCHIVED (removed from the dropdown, still defined in the backend):
                    General:       "EU Parliament", "Arab League summit", "diplomatic meeting"
                    Simultaneous:  "live TV broadcast"
                    Sight transl.: "public service consultation"
                    Institutions:  "UN Security Council", "ECOSOC", "WHO", "ILO",
                                   "UNESCO", "OIF", "AUF"
                  Their EN/AR/FR labels (scenEUParl, scenArabLeague, scenDiplomatic,
                  scenLiveTV, scenPublicService, scenSecurityCouncil, scenEcosoc,
                  scenWho, scenIlo, scenUnesco, scenOif, scenAuf) are also kept. */}
              <SelectField label={labels.scenarioLabel || 'Speaker style / setting'} id="f-scenario" name="scenario" value={form.scenario} onChange={updateField}>
                <optgroup label={labels.scenGroupGeneral || 'General / conference'}>
                  <option value="UN General Assembly">{labels.scenUNGA || 'Conference / institutional speech'}</option>
                  <option value="press conference">{labels.scenPress || 'Press conference'}</option>
                  <option value="political debate">{labels.scenDebate || 'Political debate'}</option>
                  <option value="interview">{labels.scenInterview || 'Interview'}</option>
                  <option value="panel discussion">{labels.scenPanel || 'Panel discussion'}</option>
                </optgroup>
                <optgroup label={labels.scenGroupConsec || 'Community interpreting'}>
                  <option value="legal/court setting">{labels.scenLegalCourt || 'Legal/Court setting'}</option>
                  <option value="medical/healthcare">{labels.scenMedical || 'Medical/Healthcare'}</option>
                </optgroup>
              </SelectField>
              <SelectField label={labels.structure} id="f-structure" name="structure" value={form.structure} onChange={updateField}>
                <option value="well-organized">{labels.optWellOrganized || 'Well organized'}</option>
                <option value="semi-structured">{labels.optSemiStructured || 'Semi-structured'}</option>
                <option value="deliberately disorganized">{labels.optDisorganizedFull || 'Deliberately disorganized'}</option>
              </SelectField>
              <SelectField label={labels.numbers} id="f-numbers" name="number_density" value={form.number_density} onChange={updateField}>
                <option value="low">{labels.optLowNumbers || 'Low numbers'}</option>
                <option value="medium">{labels.optTermMedium || 'Medium'}</option>
                <option value="high">{labels.optHighNumbers || 'High numbers'}</option>
              </SelectField>
              <SelectField label={labels.termDensity || 'Terminology density'} id="f-term" name="terminology_density" value={form.terminology_density} onChange={updateField}>
                <option value="low">{labels.optTermLow || 'Low — everyday vocabulary'}</option>
                <option value="medium">{labels.optTermMedium || 'Medium'}</option>
                <option value="high">{labels.optTermHigh || 'High — dense specialised terms'}</option>
              </SelectField>
              <SelectField label={labels.speedPressure || 'Speed pressure'} id="f-speed" name="speed_pressure" value={form.speed_pressure} onChange={updateField}>
                <option value="normal">{labels.optNormalSpeed || 'Normal speed'}</option>
                <option value="fast">{labels.optFast || 'Fast'}</option>
                <option value="very_fast">{labels.optVeryFast || 'Very fast'}</option>
              </SelectField>
              <SelectField label={labels.topicShifts || 'Topic shifts'} id="f-shifts" name="topic_shifts" value={form.topic_shifts} onChange={updateField}>
                <option value="none">{labels.optNoShifts || 'No topic shifts'}</option>
                <option value="mild">{labels.optShifts || 'Topic shifts'}</option>
              </SelectField>
              <div className="field" style={{ display: 'flex', alignItems: 'flex-end' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontWeight: 500 }}>
                  <input
                    type="checkbox"
                    name="include_hesitations"
                    checked={form.include_hesitations}
                    onChange={updateField}
                  />
                  {labels.hesitationsSim || 'Simulate hesitations'}
                </label>
              </div>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--warm-gray)', marginTop: '0.75rem', lineHeight: 1.5 }}>
              💡 {labels.diffHint}
            </p>
          </div>
        )}

        {/* ── Own text, no AI: paste a real text and only have it read aloud ── */}
        {ownTextMode && (
          <div className="advanced-panel" style={{ marginTop: '0.75rem' }}>
            <p style={{ fontWeight: 700, fontSize: '0.9rem', margin: 0 }}>{labels.ownTextTitle}</p>
            <p style={{ fontSize: '0.82rem', color: 'var(--warm-gray)', margin: '0.35rem 0 0.6rem', lineHeight: 1.55 }}>
              {labels.ownTextHint}
            </p>
            <textarea
              value={ownText}
              onChange={e => setOwnText(e.target.value)}
              rows={8}
              placeholder={labels.ownTextPlaceholder}
              className={form.language === 'ar' ? 'arabic' : ''}
              dir={form.language === 'ar' ? 'rtl' : 'ltr'}
              style={{ width: '100%', resize: 'vertical', padding: '0.6rem', borderRadius: 8,
                       border: '1px solid var(--border, #ddd)', fontFamily: 'inherit', fontSize: '0.9rem' }}
            />
            <div style={{ display: 'flex', gap: '0.6rem', marginTop: '0.6rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <button type="button" className="btn-primary" onClick={handleUseOwnText}>
                {labels.ownTextUse}
              </button>
              <span style={{ fontSize: '0.78rem', color: 'var(--warm-gray)' }}>
                {ownText.trim().split(/\s+/).filter(Boolean).length} {labels.wordsUnit}
              </span>
            </div>
          </div>
        )}

        {/* ── Action buttons ── */}
        <div className="action-row">
          <button type="button" className="btn-un-library" onClick={() => setShowLibrary(true)} disabled={isLoading}>
            📚 {labels.addSource || 'Add source'}
          </button>
          {/* Bring your own text — audio only, the AI never rewrites it. */}
          <button type="button" className={ownTextMode ? 'btn-primary' : 'btn-secondary'}
            onClick={() => setOwnTextMode(v => !v)} disabled={isLoading}>
            {labels.ownTextBtn}
          </button>
          {/* Upload a real speech (audio/video) to interpret from (Mariam
              feedback) — it is transcribed and used as the source. */}
          <button type="button" className="btn-secondary" onClick={() => sourceMediaRef.current?.click()}
            disabled={isLoading || sourceMediaStatus === 'loading' || sourceMediaStatus === 'materials'}>
            🎬 {sourceMediaStatus === 'loading' ? (labels.transcribing || 'Transcribing…')
              : sourceMediaStatus === 'materials' ? (labels.generatingMaterials || 'Generating materials…')
              : (labels.uploadSourceMedia || 'Upload a speech to interpret')}
          </button>
          <input ref={sourceMediaRef} type="file" accept="audio/*,video/*" style={{ display: 'none' }}
            onChange={e => { const f = e.target.files?.[0]; if (f) handleSourceMediaUpload(f); e.target.value = ''; }} />
          {hasSources ? (
            <>
              <button type="button" className="btn-primary" onClick={handleDocumentGenerate} disabled={isLoading}>
                {isLoading ? labels.generating : `▷ ${labels.generateFromDocument || 'Generate from document'}`}
              </button>
            </>
          ) : (
            <button type="submit" className="btn-primary" disabled={isLoading}>
              {isLoading ? labels.generating : `▷ ${labels.submit || 'Generate speech'}`}
            </button>
          )}
        </div>

      </form>

      {error && <div className="error-msg" style={{ marginTop: '0.75rem' }}>{labels.errorPrefix}: {error}</div>}
      {isLoading && (
        <div className="card" style={{ textAlign: 'center', padding: '2rem', marginTop: '1rem' }}>
          <div className="spinner" />
          <p style={{ marginTop: '1rem', color: 'var(--warm-gray)' }}>{labels.generating}</p>
        </div>
      )}
      {/* Show the just-generated speech, OR a restored session's speech text.
          Before, this read only the local `result`, so after "Continue" the
          summary/glossary came back (Module B reads lastGeneratedScript) but the
          SPEECH TEXT stayed blank — the most important part. Fall back to the
          restored script (user report 12 Aug 2026). */}
      {(result || lastGeneratedScript?.script) &&
        <SpeechResult data={result || lastGeneratedScript} labels={labels}
          onSaveEdit={handleSaveScriptEdit} editStatus={editStatus} />}

      {showLibrary && (
        <SourcesPanel
          labels={labels}
          language={form.language}
          domain={form.domain}
          initialQuery={form.topic}
          onSelectLibrary={src => { setLibrarySources(prev => [...prev, src]); setResult(null); setError(''); }}
          onSelectFile={file => { setDocumentFiles(prev => [...prev, file]); setResult(null); setError(''); }}
          onClose={() => setShowLibrary(false)}
        />
      )}
    </div>
  );
}

function SelectField({ label, id, name, value, onChange, children }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <select id={id} name={name} value={value} onChange={onChange}>
        {children}
      </select>
    </div>
  );
}

function SummaryBlock({ text, isArabic }) {
  const lines = (text || '').split('\n');
  return (
    <div className={`summary-block ${isArabic ? 'arabic' : ''}`}>
      {lines.map((line, i) => {
        const trimmed = line.trim();
        if (!trimmed) return null;
        if (/^[IVXLCDM]+[.)]\s/.test(trimmed)) {
          return <div key={i} className="summary-heading">{trimmed}</div>;
        }
        if (/^[-•*]\s/.test(trimmed)) {
          return <div key={i} className="summary-bullet">{trimmed.replace(/^[-•*]\s/, '')}</div>;
        }
        if (/^\d+[.)]\s/.test(trimmed)) {
          return <div key={i} className="summary-subpoint">{trimmed}</div>;
        }
        return <div key={i} className="summary-line">{trimmed}</div>;
      })}
    </div>
  );
}

function SpeechResult({ data, labels, onSaveEdit, editStatus }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const duration = useMemo(() => {
    const seconds = Number(data.estimated_duration_seconds || 0);
    if (!seconds) return null;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  }, [data.estimated_duration_seconds]);

  const isArabic = data.language === 'ar';
  const requestedRange = data.word_count_range?.label;
  const rangeMissed = data.word_count_range && data.word_count_range.within_range === false;
  const topicLooksLikeDomain = String(data.topic || '').trim().toLowerCase()
    === String(data.domain || '').trim().toLowerCase();
  const visibleDomain = (!['groq', 'openai', 'gemini', 'local_aya', 'remote_aya'].includes(
    String(data.domain || '').trim().toLowerCase()
  ) && !topicLooksLikeDomain) ? data.domain : '';

  return (
    <div className="speech-result">
      <div className="result-meta">
        <span>
          {data.word_count} {labels.wordsUnit}
          {requestedRange ? ` / ${requestedRange}` : ''}
          {rangeMissed ? ` ${labels.outsideRange}` : ''}
        </span>
        {duration && <span>~{duration}</span>}
        {data.topic && <span>{data.topic}</span>}
        {visibleDomain && <span>{visibleDomain}</span>}
        <span>{String(data.language || '').toUpperCase()} → {String(data.target_language || '').toUpperCase()}</span>
        {data.ai_generated === false && (
          <span style={{ color: 'var(--sage)', fontWeight: 600 }}>✍️ {labels.ownTextBadge}</span>
        )}
      </div>
      {/* The text came from automatic transcription of an uploaded file — say
          how reliable that is instead of presenting it as a clean source
          (ETIB question 21 Aug 2026: "dans quelle mesure la retranscription
          est-elle fiable ?"). */}
      {data.source_upload && (
        <div className="info-tip" style={{ marginBottom: '0.6rem' }}>ℹ️ {labels.mediaAsrWarning}</div>
      )}
      {data.mode === 'un_library_grounded' && data.source_speech && (
        <p className="grounded-source-note">
          {labels.groundedSourceLabel}{' '}
          {data.source_speech.source_label ? `[${data.source_speech.source_label}] ` : ''}
          {data.source_speech.web_url ? (
            <a href={data.source_speech.web_url} target="_blank" rel="noopener noreferrer">
              {data.source_speech.title || data.source_speech.un_id}
            </a>
          ) : (
            data.source_speech.title || data.source_speech.un_id
          )}
          {data.source_speech.date ? ` (${data.source_speech.date})` : ''}
        </p>
      )}
      {/* The speech is corrected HERE, at the source (user request 21 Aug 2026).
          Editing it re-derives the summary, MCQs and glossary from the edited
          text, so changing a figure or half the speech no longer leaves the
          materials describing the old version. */}
      {onSaveEdit && (
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center', margin: '0.25rem 0 0.6rem' }}>
          {!editing ? (
            <button type="button" className="btn-secondary btn-sm"
              onClick={() => { setDraft(data.script || ''); setEditing(true); }}>
              {labels.editScriptBtn}
            </button>
          ) : (
            <>
              <button type="button" className="btn-primary btn-sm"
                onClick={() => { setEditing(false); onSaveEdit(draft); }}>
                {labels.editScriptDone}
              </button>
              <button type="button" className="btn-secondary btn-sm" onClick={() => setEditing(false)}>
                {labels.aiDisclaimerLess}
              </button>
              {data.script_original && (
                <button type="button" className="btn-secondary btn-sm"
                  onClick={() => setDraft(data.script_original)}>
                  {labels.scriptRestore}
                </button>
              )}
              <span style={{ fontSize: '0.76rem', color: 'var(--warm-gray)' }}>{labels.editScriptHint}</span>
            </>
          )}
          {editStatus === 'loading' && (
            <span style={{ fontSize: '0.78rem', color: 'var(--warm-gray)' }}>⏳ {labels.materialsGenerating}</span>
          )}
        </div>
      )}
      {editing ? (
        <textarea
          value={draft}
          onChange={e => setDraft(e.target.value)}
          rows={16}
          className={isArabic ? 'arabic' : ''}
          dir={isArabic ? 'rtl' : 'ltr'}
          style={{ width: '100%', resize: 'vertical', padding: '0.7rem', borderRadius: 8,
                   border: '1px solid var(--border, #ddd)', fontFamily: 'inherit',
                   fontSize: '0.95rem', lineHeight: 1.7 }}
        />
      ) : (
        <div className={`speech-text ${isArabic ? 'arabic' : ''}`}>
          {data.script}
        </div>
      )}
      <p className="speech-next-hint">{labels.nextHintPre} <strong>{labels.navB}</strong> {labels.nextHintPost}</p>
    </div>
  );
}

// ── Interactive MCQ quiz — answers hidden until the student answers ─────────

function McqQuiz({ mcqs, labels, isArabic }) {
  const [answers, setAnswers] = useState({});

  function getCorrectText(item) {
    if (item.answer_text) return item.answer_text;
    if (Number.isInteger(item.answer_index) && Array.isArray(item.options))
      return item.options[item.answer_index] || '';
    const letter = String(item.answer || '').trim().toUpperCase();
    const idx = { A: 0, B: 1, C: 2, D: 3 }[letter];
    if (idx !== undefined && Array.isArray(item.options)) return item.options[idx] || '';
    return item.answer || '';
  }

  function isCorrectOption(item, option) {
    if (Number.isInteger(item.answer_index) && Array.isArray(item.options))
      return item.options[item.answer_index] === option;
    return option === getCorrectText(item);
  }

  function select(qi, option) {
    setAnswers(prev => (prev[qi]?.revealed ? prev : { ...prev, [qi]: { selected: option, revealed: false } }));
  }

  function check(qi) {
    setAnswers(prev => ({ ...prev, [qi]: { ...(prev[qi] || {}), revealed: true } }));
  }

  const revealedCount = mcqs.filter((_, qi) => answers[qi]?.revealed).length;
  const correctCount = mcqs.filter((item, qi) =>
    answers[qi]?.revealed && answers[qi]?.selected && isCorrectOption(item, answers[qi].selected)
  ).length;

  return (
    <div>
      <p style={{ fontSize: '0.83rem', color: 'var(--warm-gray)', marginBottom: '0.9rem' }}>
        {labels.mcqPickAnswer}
        {revealedCount > 0 && (
          <strong style={{ marginInlineStart: '0.75rem' }}>
            {labels.mcqYourScore}: {correctCount}/{revealedCount}
          </strong>
        )}
      </p>
      {mcqs.map((item, qi) => {
        const state = answers[qi] || {};
        const revealed = Boolean(state.revealed);
        const selectedIsCorrect = state.selected ? isCorrectOption(item, state.selected) : false;
        const correctText = getCorrectText(item);
        return (
          <div key={qi} className="eval-item" style={{ marginBottom: '1.1rem', paddingBottom: '0.9rem', borderBottom: '1px solid var(--border, #eee)' }}>
            <div className={`mcq-q ${isArabic ? 'arabic' : ''}`} style={{ fontWeight: 600, marginBottom: '0.55rem' }}>
              {qi + 1}. {item.question}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              {(item.options || []).map((opt, oi) => {
                const letter = 'ABCD'[oi] || String(oi + 1);
                const isSelected = state.selected === opt;
                const optionCorrect = isCorrectOption(item, opt);
                let border = '1.5px solid var(--border, #ddd)';
                let background = 'transparent';
                if (revealed && optionCorrect) { border = '1.5px solid #2D5A4E'; background = '#2d5a4e14'; }
                else if (revealed && isSelected && !optionCorrect) { border = '1.5px solid #8B3A2A'; background = '#8b3a2a14'; }
                else if (isSelected) { border = '1.5px solid var(--primary, #1a3a5c)'; background = '#1a3a5c0d'; }
                return (
                  <button
                    key={oi}
                    type="button"
                    onClick={() => select(qi, opt)}
                    className={isArabic ? 'arabic' : ''}
                    style={{
                      textAlign: 'start', padding: '0.5rem 0.75rem', borderRadius: 8,
                      border, background, cursor: revealed ? 'default' : 'pointer',
                      fontSize: '0.86rem', lineHeight: 1.45,
                      display: 'flex', gap: '0.55rem', alignItems: 'flex-start',
                    }}
                  >
                    <span style={{ fontWeight: 700, minWidth: '1.1rem', color: 'var(--warm-gray)' }}>{letter}.</span>
                    <span style={{ flex: 1 }}>{opt}{revealed && optionCorrect && ' ✓'}</span>
                  </button>
                );
              })}
            </div>
            {!revealed ? (
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.55rem' }}>
                <button type="button" className="btn-secondary btn-sm" onClick={() => check(qi)} disabled={!state.selected}>
                  {labels.mcqCheck}
                </button>
                <button type="button" className="btn-secondary btn-sm" style={{ opacity: 0.7 }} onClick={() => check(qi)}>
                  {labels.mcqShowAnswer}
                </button>
              </div>
            ) : (
              <p style={{ marginTop: '0.55rem', fontSize: '0.84rem', fontWeight: 600,
                color: state.selected ? (selectedIsCorrect ? '#2D5A4E' : '#8B3A2A') : 'var(--warm-gray)' }}>
                {state.selected
                  ? (selectedIsCorrect
                    ? labels.mcqCorrectMsg
                    : <>{labels.mcqWrongMsg} <strong>{correctText}</strong></>)
                  : <>{labels.mcqShowAnswerLabel} <strong>{correctText}</strong></>}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Sight translation scroller (professor request, cf. scroller.dipintra.it) ─
// Continuous vertical scroll at a words/min pace with a live dashboard
// (elapsed / remaining), like the University of Bologna Scroller tool.

function SightScroller({ script, isArabic, labels, isRecording, isStarting = false, onToggleRecord, recordTime }) {
  const [playing, setPlaying] = useState(false);
  const [wpm, setWpm] = useState(120);
  const [fontSize, setFontSize] = useState(1.15);   // rem
  const [columns, setColumns] = useState(1);
  const [spacing, setSpacing] = useState(1.9);      // line-height
  const [progress, setProgress] = useState(0);      // 0..1 of the exercise
  const wrapRef = useRef(null);
  const rafRef = useRef(null);
  const lastTsRef = useRef(null);
  const lastUiUpdateRef = useRef(0);
  // Time-driven engine (like scroller.dipintra.it): progress is a fraction of
  // elapsed exercise TIME, and the scroll position is derived from it each
  // frame. Timer and scroll can never disagree, and the float accumulator
  // avoids the whole-pixel scrollTop rounding that froze slow speeds.
  const progressRef = useRef(0);

  const wordCount = useMemo(() => String(script || '').split(/\s+/).filter(Boolean).length, [script]);
  const totalSeconds = (wordCount / Math.max(40, wpm)) * 60;

  useEffect(() => {
    if (!playing) {
      cancelAnimationFrame(rafRef.current);
      lastTsRef.current = null;
      return;
    }
    // Re-sync with the scrollbar (the user may have dragged it while paused).
    const wrapAtStart = wrapRef.current;
    if (wrapAtStart) {
      const scrollable = wrapAtStart.scrollHeight - wrapAtStart.clientHeight;
      if (scrollable > 0 && wrapAtStart.scrollTop > 0) {
        progressRef.current = wrapAtStart.scrollTop / scrollable;
      }
    }
    function step(ts) {
      const wrap = wrapRef.current;
      if (!wrap) return;
      if (lastTsRef.current != null && totalSeconds > 0) {
        const dt = (ts - lastTsRef.current) / 1000;
        progressRef.current = Math.min(1, progressRef.current + dt / totalSeconds);
        const scrollable = wrap.scrollHeight - wrap.clientHeight;
        if (scrollable > 0) {
          wrap.scrollTop = progressRef.current * scrollable;
        }
        if (ts - lastUiUpdateRef.current > 200) {
          lastUiUpdateRef.current = ts;
          setProgress(progressRef.current);
        }
        if (progressRef.current >= 1) {
          setProgress(1);
          setPlaying(false);
        }
      }
      lastTsRef.current = ts;
      rafRef.current = requestAnimationFrame(step);
    }
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, totalSeconds]);

  function reset() {
    setPlaying(false);
    setProgress(0);
    progressRef.current = 0;
    if (wrapRef.current) wrapRef.current.scrollTop = 0;
  }

  function fmtTime(seconds) {
    const s = Math.max(0, Math.round(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }

  const elapsed = progress * totalSeconds;
  const wordsRead = Math.round(progress * wordCount);

  return (
    <div>
      <p style={{ fontSize: '0.83rem', color: 'var(--warm-gray)', marginBottom: '0.75rem' }}>{labels.scrollerHint}</p>

      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: '0.9rem' }}>
        <div className="field" style={{ minWidth: 160 }}>
          <label>{labels.scrollerSpeed}: <strong>{wpm}</strong></label>
          <input type="range" min="80" max="240" step="10" value={wpm}
            onChange={e => setWpm(Number(e.target.value))} className="rate-slider" />
        </div>
        <div className="field" style={{ minWidth: 100 }}>
          <label>{labels.scrollerFontSize}</label>
          <select value={fontSize} onChange={e => setFontSize(Number(e.target.value))}>
            <option value={0.95}>A</option>
            <option value={1.15}>A+</option>
            <option value={1.45}>A++</option>
            <option value={1.8}>A+++</option>
          </select>
        </div>
        <div className="field" style={{ minWidth: 90 }}>
          <label>{labels.scrollerColumns}</label>
          <select value={columns} onChange={e => setColumns(Number(e.target.value))}>
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </div>
        <div className="field" style={{ minWidth: 100 }}>
          <label>{labels.scrollerSpacing}</label>
          <select value={spacing} onChange={e => setSpacing(Number(e.target.value))}>
            <option value={1.55}>1</option>
            <option value={1.9}>1.5</option>
            <option value={2.4}>2</option>
          </select>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button type="button" className="btn-primary" onClick={() => setPlaying(p => !p)}>
            {playing ? labels.scrollerPause : labels.scrollerPlay}
          </button>
          <button type="button" className="btn-secondary" onClick={reset}>{labels.scrollerReset}</button>
          {/* Recording at the same level as scrolling (16 July feedback) — no
              page scroll needed between starting the text and starting to speak */}
          {onToggleRecord && (
            <button type="button" className={`btn-record ${isRecording ? 'recording-active' : ''}`}
              onClick={onToggleRecord} disabled={isStarting}>
              {isStarting ? labels.micWarmingUp
                : isRecording ? <><span className="rec-dot" /> {labels.stopBtn} — {recordTime}</>
                : labels.recordBtn}
            </button>
          )}
        </div>
      </div>

      {/* Live dashboard — elapsed / remaining / words, like the Bologna tool */}
      <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', marginBottom: '0.6rem',
        fontSize: '0.8rem', color: 'var(--warm-gray)', fontVariantNumeric: 'tabular-nums' }}>
        <span>⏱ {fmtTime(elapsed)} / {fmtTime(totalSeconds)}</span>
        <span>⏳ −{fmtTime(totalSeconds - elapsed)}</span>
        <span>📖 {wordsRead} / {wordCount} {labels.wordsUnit}</span>
        <span style={{ flex: 1, alignSelf: 'center', height: 4, background: 'rgba(27,58,107,0.12)', borderRadius: 2, minWidth: 120 }}>
          <span style={{ display: 'block', height: '100%', width: `${Math.round(progress * 100)}%`,
            background: 'var(--primary, #1a3a5c)', borderRadius: 2, transition: 'width 0.25s linear' }} />
        </span>
      </div>

      {/* Outer wrapper owns the fixed height + vertical scrollbar; the inner
          block owns the column layout with natural height — putting both on
          one element makes CSS columns overflow horizontally (no scrolling).
          Spacer divs make the text enter from the bottom and exit through the
          top, teleprompter-style, like the Bologna Scroller tool. */}
      <div
        ref={wrapRef}
        style={{
          height: 340, overflowY: 'auto', overflowX: 'hidden',
          border: '1px solid var(--border, #ddd)', borderRadius: 10,
          background: 'var(--surface, #fff)',
        }}
      >
        <div aria-hidden="true" style={{ height: 300 }} />
        <div
          className={isArabic ? 'arabic' : ''}
          dir={isArabic ? 'rtl' : 'ltr'}
          style={{
            padding: '0 1.5rem',
            fontSize: `${fontSize}rem`, lineHeight: spacing,
            columnCount: columns, columnGap: '2.5rem', whiteSpace: 'pre-wrap',
          }}
        >
          {script}
        </div>
        <div aria-hidden="true" style={{ height: 280 }} />
      </div>
    </div>
  );
}

// ── Module B — TTS + Pedagogical Materials ──────────────────────────────────

function ModuleB({ labels, lastGeneratedScript, onAudioGenerated, onScriptUpdate, isRtl }) {
  const language = lastGeneratedScript?.language || 'ar';
  const isArabic = language === 'ar';
  const voiceOptions = VOICE_OPTIONS[language] || VOICE_OPTIONS.en;
  const speechId = lastGeneratedScript?.script?.slice(0, 50) || '';

  const [selectedAccent, setSelectedAccent] = useState(voiceOptions[0]?.accent || '');
  const [speechRate, setSpeechRate] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioStatus, setAudioStatus] = useState('idle');
  const [audioError, setAudioError] = useState('');
  // Measured speaking rate of the generated audio (Prof. LSF: simultaneous
  // sources must stay around 100-120 wpm — show the real number, not a guess).
  const [audioWpm, setAudioWpm] = useState(null);
  const [editingGlossary, setEditingGlossary] = useState(false);
  const [glossaryUploading, setGlossaryUploading] = useState(false);
  // Count of remembered glossary corrections (Lina 6 Aug) — drives the hint and
  // the "forget" button. Refreshed after each edit so the count stays live.
  const [glossaryMemoryCount, setGlossaryMemoryCount] = useState(() => glossaryMemorySize());
  const [materialsStatus, setMaterialsStatus] = useState('idle');
  const glossaryFileRef = useRef(null);

  // Recovery path: if a speech came back with its script but no MCQ/glossary
  // (rare — a long generation whose materials call was rate-limited even after
  // the backend safety net), regenerate the materials on demand.
  async function handleRegenerateMaterials() {
    if (!lastGeneratedScript?.script) return;
    setMaterialsStatus('loading');
    try {
      const mats = await generateMaterialsFromScript({
        script: lastGeneratedScript.script,
        language,
        domain: lastGeneratedScript.domain || 'general',
      });
      onScriptUpdate?.({
        ...lastGeneratedScript,
        summary:  lastGeneratedScript.summary || mats.summary || '',
        mcqs:     (lastGeneratedScript.mcqs?.length ? lastGeneratedScript.mcqs : mats.mcqs) || [],
        glossary: (lastGeneratedScript.glossary?.length ? lastGeneratedScript.glossary : mats.glossary) || [],
      });
      setMaterialsStatus('idle');
    } catch (err) {
      setMaterialsStatus('error');
      alert(`${labels.errorPrefix}: ${err.message}`);
    }
  }

  // Professor bilan (23 July): let the student upload their OWN glossary file so
  // it is taken into account in the evaluation phase. Imported terms are merged
  // onto the current glossary, which already flows into Module D.
  async function handleUploadGlossary(e) {
    const file = e.target.files?.[0];
    if (e.target) e.target.value = '';   // allow re-uploading the same file
    if (!file) return;
    setGlossaryUploading(true);
    try {
      const { glossary: imported } = await uploadGlossary(file);
      if (!imported?.length) { alert(labels.glossaryUploadEmpty); return; }
      const current = lastGeneratedScript?.glossary || [];
      // Merge: skip an imported row whose term already exists (case-insensitive).
      const seen = new Set(current.map(g => String(g.term || '').trim().toLowerCase()).filter(Boolean));
      const merged = [...current];
      for (const row of imported) {
        const key = String(row.term || '').trim().toLowerCase();
        if (key && seen.has(key)) continue;
        if (key) seen.add(key);
        merged.push(row);
      }
      onScriptUpdate?.({ ...lastGeneratedScript, glossary: merged });
      setEditingGlossary(true);
    } catch (err) {
      alert(`${labels.errorPrefix}: ${err.message}`);
    } finally {
      setGlossaryUploading(false);
    }
  }

  function updateGlossaryCell(index, key, value) {
    const current = lastGeneratedScript?.glossary || [];
    // Alias keys the LLM may have used for the same field. They must be
    // dropped on edit: the display/download getters fall back through them,
    // so a stale 'ar'/'fr'/'en' value would resurface whenever the student
    // cleared or rewrote a cell (tester bug Aug 2026 — "my modifications
    // were not taken into account").
    const aliasKeys = {
      arabic:  ['Arabic', 'ar', 'AR', 'arabic_term', 'term_ar', 'arabic_translation', 'translation_ar', 'العربية', 'عربي'],
      french:  ['French', 'fr', 'FR', 'french_term', 'term_fr', 'french_translation', 'translation_fr', 'français', 'francais'],
      english: ['English', 'en', 'EN', 'english_term', 'term_en', 'english_translation', 'translation_en'],
    }[key] || [];
    const updated = current.map((row, i) => {
      if (i !== index) return row;
      const next = { ...row, [key]: value, manually_edited: true };
      for (const alias of aliasKeys) delete next[alias];
      // The student's value outranks whatever a terminology base returned, so
      // the row must stop advertising itself as confirmed by that base.
      if (GLOSSARY_MEMORY_FIELDS.includes(key)) {
        delete next.verified;
        delete next.verified_sources;
      }
      return next;
    });
    // Remember this correction so future glossaries auto-apply it (Lina 6 Aug:
    // "can the platform retain corrections so entries improve over time?").
    // We key on the term, so an edited equivalent/definition is reused whenever
    // the same term reappears in a later generation.
    // Context travels with the correction so it can be reused knowingly later:
    // which pair it was corrected for, and in which domain.
    const correctionContext = {
      source_language: lastGeneratedScript?.language || '',
      target_language: lastGeneratedScript?.target_language || '',
      domain:          lastGeneratedScript?.domain || '',
    };
    rememberGlossaryCorrection(current[index]?.term, key, value, correctionContext);
    // Persist to the user's account (server) too, so it follows them across
    // sessions/devices. No-ops for guests (no token). Fire-and-forget.
    if (GLOSSARY_MEMORY_FIELDS.includes(key) && (current[index]?.term || '').trim() && String(value || '').trim()) {
      saveGlossaryCorrection({
        term: current[index].term,
        [key]: value,
        ...correctionContext,
        manually_edited: true,
      }).catch(() => {});
    }
    setGlossaryMemoryCount(glossaryMemorySize());
    // Propagate upward so Module D evaluates terminology against the
    // student-corrected glossary (cahier des charges request).
    onScriptUpdate?.({ ...lastGeneratedScript, glossary: updated });
  }

  // Students build their own terminology base (professor feedback 20 July).
  function addGlossaryTerm() {
    const current = lastGeneratedScript?.glossary || [];
    const updated = [...current, { term: '', arabic: '', french: '', english: '', definition: '' }];
    onScriptUpdate?.({ ...lastGeneratedScript, glossary: updated });
    setEditingGlossary(true);
  }

  function removeGlossaryTerm(index) {
    const current = lastGeneratedScript?.glossary || [];
    onScriptUpdate?.({ ...lastGeneratedScript, glossary: current.filter((_, i) => i !== index) });
  }

  useEffect(() => {
    const opts = VOICE_OPTIONS[language] || VOICE_OPTIONS.en;
    setSelectedAccent(opts[0]?.accent || '');
    setAudioUrl(null);
    setAudioStatus('idle');
    setAudioError('');
    setAudioWpm(null);
  }, [speechId]);

  if (!lastGeneratedScript) {
    return (
      <div className="card coming-soon">
        <p className="eyebrow">{labels.moduleBTitle}</p>
        <p>{labels.noSpeechYet}</p>
      </div>
    );
  }

  const summary  = lastGeneratedScript.summary  || '';
  const mcqs     = lastGeneratedScript.mcqs     || [];
  const glossary = lastGeneratedScript.glossary || [];

  async function handleGenerateAudio() {
    setAudioStatus('loading'); setAudioError('');
    try {
      const data = await textToSpeech({ text: lastGeneratedScript.script, language, accent: selectedAccent, rate_adjustment: speechRate });
      const url = `${SERVER_BASE}${data.audio_url}`;
      setAudioWpm(data.words_per_minute || null);
      setAudioUrl(url); onAudioGenerated?.(url); setAudioStatus('success');
    } catch (err) { setAudioError(err.message); setAudioStatus('error'); }
  }

  async function handleDownloadGlossary() {
    try {
      const mapped = glossary.map(item => ({
        ar: glossaryArabicValue(item),
        fr: glossaryValue(item, ['french', 'French', 'fr', 'FR', 'french_term', 'term_fr', 'french_translation', 'translation_fr', 'français', 'francais']),
        en: glossaryValue(item, ['english', 'English', 'en', 'EN', 'english_term', 'term_en', 'english_translation', 'translation_en']),
      }));
      const blob = await downloadGlossary({ glossary: mapped, domain: lastGeneratedScript.domain });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url;
      a.download = `glossary_${lastGeneratedScript.domain}.docx`; a.click();
      URL.revokeObjectURL(url);
    } catch (err) { alert(`${labels.errorPrefix}: ${err.message}`); }
  }

  return (
    <div className="module-b-layout">

      {/* Materials recovery: script present but MCQ/glossary missing. */}
      {lastGeneratedScript.script && (mcqs.length === 0 || glossary.length === 0) && (
        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', background: '#fff7e6', border: '1px solid #f0d9a8' }}>
          <span style={{ flex: 1, fontSize: '0.9rem' }}>
            {lastGeneratedScript.ai_generated === false
              ? <>ℹ️ {labels.ownTextMaterialsHint}</>
              : <>⚠️ {labels.materialsMissing}</>}
          </span>
          <button className="btn-primary" onClick={handleRegenerateMaterials} disabled={materialsStatus === 'loading'} style={{ whiteSpace: 'nowrap' }}>
            {materialsStatus === 'loading' ? labels.materialsGenerating : labels.generateMaterials}
          </button>
        </div>
      )}

      {/* ── Audio ── */}
      <div className="card">
        <h2 className="b-section-title">🔊 {labels.moduleBTitle}</h2>
        <div className="form-grid" style={{ marginBottom: '1rem' }}>
          <SelectField label={labels.voiceAccent} id="b-voice" name="voice"
            value={selectedAccent} onChange={e => setSelectedAccent(e.target.value)}>
            {voiceOptions.map(v => (
              <option key={v.accent} value={v.accent}>{isRtl ? (v.labelAr || v.label) : v.label}</option>
            ))}
          </SelectField>
          <div className="field">
            <label htmlFor="b-rate">{labels.speechRate}: {speechRate > 0 ? `+${speechRate}` : speechRate}%</label>
            <input id="b-rate" type="range" min="-20" max="20" step="5"
              value={speechRate} onChange={e => setSpeechRate(Number(e.target.value))} className="rate-slider" />
          </div>
        </div>
        <button className="btn-primary" onClick={handleGenerateAudio} disabled={audioStatus === 'loading'}>
          {audioStatus === 'loading' ? labels.generatingAudio : `▷ ${labels.generateAudio}`}
        </button>
        {audioError && <div className="error-msg" style={{ marginTop: '0.75rem' }}>{labels.errorPrefix}: {audioError}</div>}
        {audioUrl && (
          <div style={{ marginTop: '1rem' }}>
            <audio key={audioUrl} controls src={audioUrl} style={{ width: '100%' }} />
            {/* Measured-pace line removed (user request 12 Aug 2026). audioWpm
                is still computed but no longer shown under the player. */}
            <a className="btn-secondary btn-sm" href={audioUrl} download={`speech_${language}.mp3`}
              style={{ display: 'inline-block', marginTop: '0.5rem' }}>
              ⬇ {labels.downloadAudio}
            </a>
          </div>
        )}
      </div>

      {/* ── Summary ── */}
      {summary && (
        <div className="card">
          <h2 className="b-section-title">📋 {labels.summaryTitle}</h2>
          <SummaryBlock text={summary} isArabic={isArabic} />
        </div>
      )}

      {/* ── MCQ — interactive: the answer stays hidden until the student answers ── */}
      {mcqs.length > 0 && (
        <div className="card">
          <h2 className="b-section-title">❓ {labels.mcqTitle}</h2>
          {/* key = speech fingerprint: a new speech mounts a fresh quiz,
              otherwise answers from the previous text stay selected */}
          <McqQuiz key={speechId} mcqs={mcqs} labels={labels} isArabic={isArabic} />
        </div>
      )}

      {/* ── Glossary — editable BEFORE recording; evaluation checks against it ── */}
      {glossary.length > 0 && (
        <div className="card">
          <div className="b-section-header">
            <h2 className="b-section-title">📖 {labels.glossaryTitle}</h2>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <button className="btn-secondary btn-sm" onClick={() => setEditingGlossary(v => !v)}>
                {editingGlossary ? labels.glossaryEditDone : labels.glossaryEdit}
              </button>
              <button className="btn-secondary btn-sm" onClick={addGlossaryTerm}>➕ {labels.glossaryAddTerm}</button>
              <button className="btn-secondary btn-sm" onClick={() => glossaryFileRef.current?.click()} disabled={glossaryUploading}>
                {glossaryUploading ? `⏳ ${labels.glossaryUploading}` : `⬆ ${labels.uploadGlossary}`}
              </button>
              <input
                ref={glossaryFileRef}
                type="file"
                accept=".csv,.tsv,.txt,.md,.docx,.json"
                onChange={handleUploadGlossary}
                style={{ display: 'none' }}
              />
              <button className="btn-secondary btn-sm" onClick={handleDownloadGlossary}>{labels.downloadGlossary}</button>
            </div>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--warm-gray)', marginBottom: '0.75rem' }}>
            💡 {labels.glossaryEditHint}
          </p>
          {glossaryMemoryCount > 0 && (
            <p style={{ fontSize: '0.8rem', color: 'var(--warm-gray)', marginBottom: '0.75rem',
                        display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
              <span>🧠 {glossaryMemoryCount} {labels.glossaryMemoryNote}</span>
              <button className="btn-secondary btn-sm" onClick={() => {
                try { localStorage.removeItem(GLOSSARY_MEMORY_KEY); } catch { /* ignore */ }
                setGlossaryMemoryCount(0);
              }}>{labels.glossaryMemoryClear}</button>
            </p>
          )}
          <div className="table-responsive">
            <table className="glossary-table">
              <thead>
                <tr>
                  <th>{labels.glossaryTermHeader || 'Term'}</th>
                  <th>{labels.glossaryArabicHeader || 'Arabic'}</th>
                  <th>{labels.glossaryFrenchHeader || 'French'}</th>
                  <th>{labels.glossaryEnglishHeader || 'English'}</th>
                  <th>{labels.glossaryDefinitionHeader || 'Definition'}</th>
                  {editingGlossary && <th aria-label="delete"></th>}
                </tr>
              </thead>
              <tbody>
                {glossary.map((item, i) => (
                  <tr key={i}>
                    {editingGlossary ? (
                      <>
                        <td><input className="glossary-edit-input" value={item.term || ''} onChange={e => updateGlossaryCell(i, 'term', e.target.value)} /></td>
                        <td><input className="glossary-edit-input arabic" dir="rtl" value={glossaryEditValue(item, 'arabic', ['Arabic', 'ar', 'AR', 'arabic_term', 'term_ar', 'arabic_translation', 'translation_ar', 'العربية', 'عربي'])} onChange={e => updateGlossaryCell(i, 'arabic', e.target.value)} /></td>
                        <td><input className="glossary-edit-input" value={glossaryEditValue(item, 'french', ['French', 'fr', 'FR', 'french_term', 'term_fr', 'french_translation', 'translation_fr', 'français', 'francais'])} onChange={e => updateGlossaryCell(i, 'french', e.target.value)} /></td>
                        <td><input className="glossary-edit-input" value={glossaryEditValue(item, 'english', ['English', 'en', 'EN', 'english_term', 'term_en', 'english_translation', 'translation_en'])} onChange={e => updateGlossaryCell(i, 'english', e.target.value)} /></td>
                        <td><input className="glossary-edit-input" value={item.definition || ''} onChange={e => updateGlossaryCell(i, 'definition', e.target.value)} /></td>
                        <td><button type="button" className="file-chip-remove" onClick={() => removeGlossaryTerm(i)} title={labels.removeSource}>×</button></td>
                      </>
                    ) : (
                      <>
                        <td>
                          <strong>{item.term}</strong>
                          {item.manually_edited ? (
                            <span title={labels.termEditedTitle}
                              style={{ marginInlineStart: '0.35rem', color: 'var(--primary)', fontSize: '0.72rem', whiteSpace: 'nowrap' }}>
                              ✎ {labels.termEditedTitle}
                            </span>
                          ) : item.verified ? (
                            <span title={`${labels.termVerifiedTitle}: ${(item.verified_sources || []).join(', ')}`}
                              style={{ marginInlineStart: '0.35rem', color: 'var(--sage)', fontSize: '0.72rem', whiteSpace: 'nowrap' }}>
                              ✓ {(item.verified_sources || [])[0] || ''}
                            </span>
                          ) : null}
                        </td>
                        <td className="arabic" dir="rtl">{glossaryArabicValue(item)}</td>
                        <td>{glossaryValue(item, ['french', 'French', 'fr', 'FR', 'french_term', 'term_fr', 'french_translation', 'translation_fr', 'français', 'francais'])}</td>
                        <td>{glossaryValue(item, ['english', 'English', 'en', 'EN', 'english_term', 'term_en', 'english_translation', 'translation_en'])}</td>
                        <td className="gloss-def">{item.definition || ''}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
}

// ── Note-taking space for consecutive interpretation (text + sketch) ─────────

function NotesPad({ labels, sourceText, sourceIsArabic, hasReferenceAudio = false, sourceProgress = 0 }) {
  const [tab, setTab] = useState('text');   // 'text' | 'sketch'
  const [notes, setNotes] = useState('');
  const [showSource, setShowSource] = useState(false);   // hidden until student chooses to reveal
  // Live transcript: the source speech transcription fills in progressively,
  // synced to the audio, beside a blank notes space (professor bilan 23 July).
  const [liveTranscript, setLiveTranscript] = useState(false);
  const [penColor, setPenColor] = useState('#1a3a5c');
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef(null);

  function canvasPos(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    // Pointer events cover mouse, touch, AND stylus natively (professor
    // feedback: stylus support for handwritten consecutive notes).
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
      pressure: e.pressure && e.pressure > 0 ? e.pressure : 0.5,
    };
  }

  function startDraw(e) {
    e.preventDefault();
    canvasRef.current?.setPointerCapture?.(e.pointerId);
    drawingRef.current = true;
    lastPointRef.current = canvasPos(e);
  }

  function draw(e) {
    if (!drawingRef.current) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const point = canvasPos(e);
    ctx.strokeStyle = penColor;
    // Stylus pressure drives line width when available; mouse stays constant.
    ctx.lineWidth = 1 + point.pressure * 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPointRef.current = point;
  }

  function endDraw() {
    drawingRef.current = false;
  }

  function clearAll() {
    if (tab === 'text') {
      setNotes('');
    } else {
      const canvas = canvasRef.current;
      canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  // Progressive source transcript, revealed sentence by sentence in sync with
  // the source audio, so the student sees the speech "being transcribed" as it
  // plays while they take notes beside it (professor bilan 23 July).
  const sourceSentences = useMemo(() => {
    if (!sourceText) return [];
    return sourceText.match(/[^.!?؟\n]+[.!?؟]*\s*/g) || [sourceText];
  }, [sourceText]);
  const revealCount = sourceProgress <= 0
    ? 0
    : Math.min(sourceSentences.length, Math.ceil(sourceProgress * sourceSentences.length));
  const revealedSource = sourceSentences.slice(0, revealCount).join('');

  const liveRef = useRef(null);
  useEffect(() => {
    if (liveTranscript && liveRef.current) liveRef.current.scrollTop = liveRef.current.scrollHeight;
  }, [revealedSource, liveTranscript]);

  const noteArea = tab === 'text' ? (
    <textarea
      value={notes}
      onChange={e => setNotes(e.target.value)}
      placeholder={labels.notesPlaceholder}
      style={{
        width: '100%', minHeight: 260, padding: '0.75rem 1rem',
        border: '1px solid var(--border, #ddd)', borderRadius: 10,
        fontSize: '0.92rem', lineHeight: 1.6, resize: 'vertical',
        fontFamily: 'inherit', background: 'var(--surface, #fff)',
      }}
    />
  ) : (
    <canvas
      ref={canvasRef}
      width={900}
      height={520}
      style={{
        width: '100%', height: 300, border: '1px solid var(--border, #ddd)',
        borderRadius: 10, background: 'var(--surface, #fff)', touchAction: 'none', cursor: 'crosshair',
      }}
      onPointerDown={startDraw} onPointerMove={draw} onPointerUp={endDraw}
      onPointerLeave={endDraw} onPointerCancel={endDraw}
    />
  );

  return (
    <div className="record-section" style={{ marginTop: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        <p className="record-section-label" style={{ margin: 0 }}>{labels.notesTitle}</p>
        <div style={{ display: 'flex', gap: '0.35rem' }}>
          <button type="button" className={`lib-tab ${tab === 'text' ? 'lib-tab-active' : ''}`} onClick={() => setTab('text')}>
            {labels.notesTextTab}
          </button>
          <button type="button" className={`lib-tab ${tab === 'sketch' ? 'lib-tab-active' : ''}`} onClick={() => setTab('sketch')}>
            {labels.notesSketchTab}
          </button>
        </div>
        {tab === 'sketch' && (
          <div style={{ display: 'flex', gap: '0.3rem', alignItems: 'center' }}>
            {['#1a3a5c', '#8B3A2A', '#2D5A4E', '#A67C1B'].map(c => (
              <button key={c} type="button" onClick={() => setPenColor(c)} aria-label={`pen ${c}`}
                style={{ width: 20, height: 20, borderRadius: '50%', background: c, cursor: 'pointer',
                  border: penColor === c ? '2px solid #000' : '1px solid #ccc' }} />
            ))}
          </div>
        )}
        {hasReferenceAudio && sourceText && (
          <button type="button" className={`btn-secondary btn-sm ${liveTranscript ? 'lib-tab-active' : ''}`}
            onClick={() => { setLiveTranscript(v => !v); if (!liveTranscript) setShowSource(false); }}>
            {liveTranscript ? labels.liveTranscriptOff : `🎙 ${labels.liveTranscriptOn}`}
          </button>
        )}
        {sourceText && !liveTranscript && (
          <button type="button" className="btn-secondary btn-sm" onClick={() => setShowSource(v => !v)}>
            {showSource ? labels.notesHideSource : labels.notesShowSource}
          </button>
        )}
        <button type="button" className="btn-secondary btn-sm" onClick={clearAll}>{labels.notesClear}</button>
      </div>

      {/* Live transcription split screen: the source speech transcription fills
          in progressively as the audio plays (left), beside a blank notes space
          (right) — professor bilan 23 July. */}
      {liveTranscript && sourceText ? (
        <div className="consec-split">
          <div className="consec-split-source">
            <p className="consec-split-label">
              🎙 {labels.liveTranscriptTitle} · {Math.round(sourceProgress * 100)}%
            </p>
            <div ref={liveRef} className={`consec-source-text ${sourceIsArabic ? 'arabic' : ''}`}
              dir={sourceIsArabic ? 'rtl' : 'ltr'} style={{ maxHeight: 300, overflowY: 'auto' }}>
              {revealedSource}
              {sourceProgress > 0 && sourceProgress < 1 && <span className="live-caret">▍</span>}
              {sourceProgress === 0 && (
                <span style={{ color: 'var(--warm-gray)' }}>{labels.liveTranscriptWaiting}</span>
              )}
            </div>
          </div>
          <div className="consec-split-notes">
            <p className="consec-split-label">✍️ {labels.notesTitle}</p>
            {noteArea}
          </div>
        </div>
      ) : showSource && sourceText ? (
        /* Static reveal: full source text beside the note area, for reviewing
           the source while taking notes without waiting for playback. */
        <div className="consec-split">
          <div className="consec-split-source">
            <p className="consec-split-label">📄 {labels.sourceText || 'Source'}</p>
            <div className={`consec-source-text ${sourceIsArabic ? 'arabic' : ''}`} dir={sourceIsArabic ? 'rtl' : 'ltr'}>
              {sourceText}
            </div>
          </div>
          <div className="consec-split-notes">
            <p className="consec-split-label">✍️ {labels.notesTitle}</p>
            {noteArea}
          </div>
        </div>
      ) : noteArea}
      <p style={{ fontSize: '0.75rem', color: 'var(--warm-gray)', marginTop: '0.4rem' }}>
        {liveTranscript ? labels.liveTranscriptHint : labels.notesHint}
      </p>
    </div>
  );
}

// ── Module C — ASR Transcription + Browser Recording ────────────────────────

function ModuleC({ labels, referenceAudioUrl, sourceScript, targetLanguage, glossary = [], onTranscriptComplete, onRecordingComplete }) {
  const [language, setLanguage] = useState('ar');

  // The student interprets INTO the target language chosen in Module A —
  // preselect it so the recognizer transcribes in the right language.
  useEffect(() => {
    if (targetLanguage && ['ar', 'fr', 'en'].includes(targetLanguage)) {
      setLanguage(targetLanguage);
    }
  }, [targetLanguage]);
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  // The microphone/encoder needs a moment before it actually captures audio.
  // Until it does we must NOT tell the student to speak and must NOT start the
  // source in simultaneous mode — two testers lost the beginning of their
  // performance that way (ETIB feedback 21 Aug 2026).
  const [isStarting, setIsStarting] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [autoTranscribe, setAutoTranscribe] = useState(true);
  // Correct the ASR output BEFORE the evaluation (ETIB feedback 21 Aug 2026:
  // "est-il possible d'éditer [le texte] après la retranscription (STT) avant
  // l'évaluation ?"). A recognition error must never be scored as an
  // interpreting error. Only the TEXT is corrected — pauses, hesitations and
  // speech rate keep being measured from the real recording server-side.
  const [editingTranscript, setEditingTranscript] = useState(false);
  const [transcriptDraft, setTranscriptDraft] = useState('');

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const referenceAudioRef = useRef(null);
  const autoTranscribeRef = useRef(autoTranscribe);
  useEffect(() => { autoTranscribeRef.current = autoTranscribe; }, [autoTranscribe]);
  const [simultaneousActive, setSimultaneousActive] = useState(false);
  const [simulSourceEnded, setSimulSourceEnded] = useState(false);
  const [simulShowText, setSimulShowText] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const graceTimerRef = useRef(null);
  // Playback progress (0..1) of the source audio — drives the live source
  // transcript that fills in beside the notes area in consecutive mode
  // (professor bilan 23 July: split screen with the speech transcription on
  // one side and a blank notes space on the other).
  const [sourceProgress, setSourceProgress] = useState(0);

  // Keep the chosen playback speed applied whenever the source audio (re)loads.
  useEffect(() => {
    if (referenceAudioRef.current) referenceAudioRef.current.playbackRate = playbackSpeed;
  }, [referenceAudioUrl, playbackSpeed]);
  // Reset the live-transcript progress whenever the source audio changes.
  useEffect(() => { setSourceProgress(0); }, [referenceAudioUrl]);

  function handleSourceTimeUpdate(e) {
    const el = e.currentTarget;
    if (el?.duration > 0) setSourceProgress(Math.min(1, el.currentTime / el.duration));
  }
  // 3 practice modes (professor request): simultaneous | consecutive | sight
  const [interpMode, setInterpMode] = useState('consecutive');
  const sourceIsArabic = /[؀-ۿ]/.test(sourceScript || '');

  const lowConfWords = useMemo(() =>
    (result?.segments || []).flatMap(seg =>
      (seg.words || []).filter(w => w.probability < 0.6)
    ), [result]);

  function formatTime(s) {
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }

  async function runTranscription(audioInput) {
    setStatus('loading');
    setError('');
    setResult(null);
    try {
      const fileToSend = audioInput instanceof File
        ? audioInput
        : new File([audioInput], 'recording.webm', { type: audioInput.type });
      const data = await transcribeAudio(fileToSend, language, sourceScript || '');
      if (data.no_speech_detected || !data.full_text?.trim()) {
        setError(language === 'ar'
          ? 'لم يُكتشف كلام — تحقق من الميكروفون وأعد المحاولة.'
          : language === 'fr'
          ? 'Aucune parole détectée — vérifiez le microphone et réessayez.'
          : 'No speech detected — check your microphone and try again.');
        setStatus('error');
        return;
      }
      setResult(data);
      setEditingTranscript(false);
      // interpMode travels with the transcript so Module D can evaluate with
      // mode-aware criteria (décalage tolerance for simultaneous, etc.) —
      // professor feedback: pauses from normal EVS should not be penalized.
      onTranscriptComplete?.({ ...data, language, sourceScript: sourceScript || '', mode: interpMode });
      setStatus('success');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  }

  // Simultaneous interpretation: the source speech plays WHILE the student
  // records — echo cancellation reduces source bleed; headphones recommended.
  async function startSimultaneous() {
    const player = referenceAudioRef.current;
    if (!player) return;
    try {
      player.currentTime = 0;
      player.playbackRate = playbackSpeed;
      setSimultaneousActive(true);
      setSimulSourceEnded(false);
      const started = await startRecording({ echoCancellation: true, noiseSuppression: true });
      if (!started) { setSimultaneousActive(false); return; }
      await player.play();
      // The interpreter necessarily finishes AFTER the speaker (décalage) —
      // when the source ends, keep recording for up to 60s so the student can
      // complete the interpretation; they can stop earlier manually.
      player.onended = () => {
        setSimulSourceEnded(true);
        graceTimerRef.current = setTimeout(() => stopSimultaneous(), 60000);
      };
    } catch (err) {
      setSimultaneousActive(false);
      setError(labels.simultaneousFailed);
    }
  }

  function stopSimultaneous() {
    const player = referenceAudioRef.current;
    if (player) { player.pause(); player.onended = null; }
    clearTimeout(graceTimerRef.current);
    setSimulSourceEnded(false);
    setSimultaneousActive(false);
    stopRecording();
  }

  async function startRecording(audioConstraints) {
    try {
      // Guard: when used as onClick handler the first arg is the click event.
      const isConstraintObject = audioConstraints && typeof audioConstraints === 'object'
        && !('target' in audioConstraints);
      const constraints = { audio: isConstraintObject ? audioConstraints : true };
      if (!navigator.mediaDevices?.getUserMedia) {
        setError(labels.micDenied);
        return false;
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus' : 'audio/webm';
      const mr = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      setRecordingTime(0);
      setResult(null);
      setError('');

      // The first chunk only arrives once the encoder is really capturing.
      // Resolving on it is what lets us gate the "you may speak" signal.
      let firstChunkSeen;
      const firstChunk = new Promise(resolve => { firstChunkSeen = resolve; });
      mr.ondataavailable = e => {
        if (e.data.size > 0) { chunksRef.current.push(e.data); firstChunkSeen(); }
      };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        stream.getTracks().forEach(t => t.stop());
        clearInterval(timerRef.current);
        setRecordedBlob(blob);
        onRecordingComplete?.(blob);
        if (recordedUrl) URL.revokeObjectURL(recordedUrl);
        setRecordedUrl(URL.createObjectURL(blob));
        if (autoTranscribeRef.current) runTranscription(blob);
      };

      setIsStarting(true);
      mr.start(250);
      // Wait for audio to be genuinely captured before switching the UI to
      // "Recording" (and, in simultaneous mode, before the source starts).
      // Bounded so a slow or non-conforming browser can never hang the button.
      await Promise.race([firstChunk, new Promise(resolve => setTimeout(resolve, 1500))]);
      // Small extra margin so the very first syllable is inside the file.
      await new Promise(resolve => setTimeout(resolve, 250));
      setIsStarting(false);
      setIsRecording(true);
      timerRef.current = setInterval(() => setRecordingTime(t => t + 1), 1000);
      return true;
    } catch (err) {
      setIsStarting(false);
      setError(labels.micDenied);
      return false;
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
    setIsStarting(false);
    clearInterval(timerRef.current);
  }

  function startEditingTranscript() {
    setTranscriptDraft(result?.full_text || '');
    setEditingTranscript(true);
  }

  function saveTranscriptEdit() {
    const text = transcriptDraft.trim();
    if (text && result && text !== result.full_text) {
      const corrected = {
        ...result,
        full_text: text,
        asr_full_text: result.asr_full_text ?? result.full_text,
        edited_by_student: true,
        // The vocalised (tashkeel) hypothesis was derived from the ASR text and
        // no longer matches the corrected one — drop it rather than display a
        // version that contradicts what the student just wrote.
        vocalized_text: '',
      };
      setResult(corrected);
      onTranscriptComplete?.({ ...corrected, language, sourceScript: sourceScript || '', mode: interpMode });
    }
    setEditingTranscript(false);
  }

  function clearTranscript() {
    setResult(null);
    setError('');
    setEditingTranscript(false);
  }

  function clearAudio() {
    if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    setRecordedBlob(null);
    setRecordedUrl(null);
    setResult(null);
    setError('');
    onRecordingComplete?.(null);
  }

  const isArabic = result?.language_detected === 'ar' || language === 'ar';

  return (
    <div>
      <div className="card">
        <h2>{labels.moduleCTitle}</h2>

        {language === 'ar' && (
          <div className="info-tip">💡 {labels.arabicTip}</div>
        )}

        {/* Language selector */}
        <div style={{ marginBottom: '1.25rem', maxWidth: 280 }}>
          <SelectField label={labels.language} id="c-lang" name="c-lang"
            value={language} onChange={e => { setLanguage(e.target.value); setResult(null); }}>
            <option value="ar">العربية - Arabic</option>
            <option value="fr">Français - French</option>
            <option value="en">English</option>
          </SelectField>
        </div>

        {/* ── Practice mode selector: simultaneous | consecutive | sight ── */}
        <div style={{ marginBottom: '1.25rem' }}>
          <p className="record-section-label" style={{ marginBottom: '0.5rem' }}>{labels.modeChoose}</p>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {[
              { id: 'simultaneous', name: labels.optSimultaneous, icon: '🎧' },
              { id: 'consecutive',  name: labels.optConsecutive,  icon: '📝' },
              { id: 'sight',        name: labels.optSight,        icon: '📜' },
            ].map(m => (
              <button
                key={m.id}
                type="button"
                onClick={() => setInterpMode(m.id)}
                style={{
                  flex: '1 1 140px', padding: '0.65rem 0.9rem', borderRadius: 10, cursor: 'pointer',
                  border: interpMode === m.id ? '2px solid var(--primary, #1a3a5c)' : '1.5px solid var(--border, #ddd)',
                  background: interpMode === m.id ? 'rgba(27,58,107,0.07)' : 'transparent',
                  fontWeight: interpMode === m.id ? 700 : 500, fontSize: '0.9rem',
                }}
              >
                {m.icon} {m.name}
              </button>
            ))}
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--warm-gray)', marginTop: '0.5rem' }}>
            {interpMode === 'simultaneous' && labels.modeSimulDesc}
            {interpMode === 'consecutive' && labels.modeConsecDesc}
            {interpMode === 'sight' && labels.modeSightDesc}
          </p>
        </div>

        {/* Reference audio from Module B — for simultaneous & consecutive */}
        {interpMode !== 'sight' && referenceAudioUrl && (
          <div className="reference-audio-box">
            <p className="ref-audio-label">🔊 {labels.sourceAudio}</p>
            <audio ref={referenceAudioRef} controls src={referenceAudioUrl} style={{ width: '100%' }}
              onTimeUpdate={handleSourceTimeUpdate} />
            {/* Playback speed for beginners (Mariam feedback: 0.75 was too big a
                jump from normal — add 0.9). Applies to this source player. */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--warm-gray)' }}>{labels.playbackSpeed}:</span>
              {[0.5, 0.75, 0.9, 1].map(sp => (
                <button key={sp} type="button"
                  className={`lib-tab ${playbackSpeed === sp ? 'lib-tab-active' : ''}`}
                  onClick={() => {
                    setPlaybackSpeed(sp);
                    if (referenceAudioRef.current) referenceAudioRef.current.playbackRate = sp;
                  }}>
                  {sp === 1 ? '1×' : `${sp}×`}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Simultaneous — source plays while the student records */}
        {interpMode === 'simultaneous' && (
          <div className="record-section" style={{ borderInlineStart: '3px solid var(--primary, #1a3a5c)', paddingInlineStart: '0.9rem', marginBottom: '1rem' }}>
            <p className="record-section-label">{labels.simulTitle}</p>
            <p style={{ fontSize: '0.8rem', color: 'var(--warm-gray)', margin: '0.3rem 0 0.6rem' }}>
              {referenceAudioUrl ? labels.simulHint : labels.simulNeedsAudio}
            </p>
            {/* "SIMUL avec texte" (20 July feedback): optionally show the source
                text on screen during the simultaneous exercise. */}
            {sourceScript && (
              <label className="auto-transcribe-toggle" style={{ marginBottom: '0.6rem' }}>
                <input type="checkbox" checked={simulShowText}
                  onChange={e => setSimulShowText(e.target.checked)} />
                {labels.simulWithText}
              </label>
            )}
            {simulShowText && sourceScript && (
              <div className={`consec-source-text ${sourceIsArabic ? 'arabic' : ''}`} dir={sourceIsArabic ? 'rtl' : 'ltr'}
                style={{ marginBottom: '0.75rem', maxHeight: 220 }}>
                {sourceScript}
              </div>
            )}
            {referenceAudioUrl && (
              !simultaneousActive ? (
                <button className="btn-primary" onClick={startSimultaneous} disabled={isRecording || isStarting}>
                  {isStarting ? labels.micWarmingUp : labels.simulStart}
                </button>
              ) : (
                <>
                  <button className="btn-record recording-active" onClick={stopSimultaneous}>
                    <span className="rec-dot" /> {labels.simulStop} — {formatTime(recordingTime)}
                  </button>
                  {simulSourceEnded && (
                    <div className="info-tip" style={{ marginTop: '0.6rem' }}>⏳ {labels.simulSourceEnded}</div>
                  )}
                </>
              )
            )}
          </div>
        )}

        {/* Sight translation — scrolling source text, record while translating */}
        {interpMode === 'sight' && (
          <div className="record-section" style={{ marginBottom: '1rem' }}>
            {sourceScript ? (
              <SightScroller key={sourceScript.slice(0, 50)} script={sourceScript} isArabic={sourceIsArabic} labels={labels}
                isRecording={isRecording}
                isStarting={isStarting}
                recordTime={formatTime(recordingTime)}
                onToggleRecord={() => (isRecording ? stopRecording() : startRecording())} />
            ) : (
              <div className="info-tip">ℹ️ {labels.needsSource}</div>
            )}
          </div>
        )}

        {/* Consecutive — pedagogical order (16 July feedback): the student
            FIRST listens and takes notes, THEN interprets from the notes.
            Split-screen note-taking (20 July feedback): source on one side,
            note space on the other. */}
        {interpMode === 'consecutive' && (
          <>
            <div className="info-tip" style={{ marginBottom: '0.75rem' }}>📝 {labels.consecFlowHint}</div>
            <NotesPad labels={labels} sourceText={sourceScript} sourceIsArabic={sourceIsArabic}
              hasReferenceAudio={!!referenceAudioUrl} sourceProgress={sourceProgress} />
          </>
        )}

        {/* Recording section */}
        <div className="record-section">
          <p className="record-section-label">🎙 {labels.yourInterpretation}</p>

          <div className="record-controls">
            {/* The record button lives here ONLY for consecutive mode.
                Simultaneous is driven by "Play source + record me" and sight by
                the scroller's own record button — showing a second "Start
                recording" here confused testers (professor feedback 20 July). */}
            {interpMode === 'consecutive' ? (
              !isRecording ? (
                <button className="btn-record" onClick={startRecording} disabled={isStarting}>
                  {isStarting ? labels.micWarmingUp : labels.recordBtn}
                </button>
              ) : (
                <button className="btn-record recording-active" onClick={stopRecording}>
                  <span className="rec-dot" /> {labels.stopBtn} — {formatTime(recordingTime)}
                </button>
              )
            ) : isRecording ? (
              <span style={{ fontSize: '0.85rem', color: 'var(--warm-gray)', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
                <span className="rec-dot" /> {labels.recordingLive}…
                {interpMode === 'simultaneous' ? ` (${labels.simulTitle})` : ''}
              </span>
            ) : (
              <span style={{ fontSize: '0.82rem', color: 'var(--warm-gray)' }}>
                {interpMode === 'simultaneous' ? labels.simulUseButton : labels.sightUseButton}
              </span>
            )}

            <label className="auto-transcribe-toggle">
              <input type="checkbox" checked={autoTranscribe}
                onChange={e => setAutoTranscribe(e.target.checked)} />
              {labels.autoTranscribe}
            </label>
          </div>
          {/* Never let the student start speaking before the encoder is live. */}
          <p style={{ fontSize: '0.78rem', color: 'var(--warm-gray)', marginTop: '0.4rem' }}>
            🎤 {labels.micReadyHint}
          </p>

          {/* Recorded audio playback */}
          {recordedUrl && !isRecording && (
            <div style={{ marginTop: '0.75rem' }}>
              <div className="recorded-audio-row">
                <audio controls src={recordedUrl} style={{ width: '100%' }} />
                {/* Recordings are deleted server-side after processing — this
                    download is the student's only copy (for sharing with peers
                    or a teacher, 16 July feedback) */}
                <a className="btn-secondary btn-sm" href={recordedUrl}
                  download={`interpretation_${new Date().toISOString().slice(0, 10)}.webm`}
                  style={{ whiteSpace: 'nowrap', alignSelf: 'center' }}>
                  ⬇ {labels.downloadRecording}
                </a>
                <button className="btn-icon-danger" onClick={clearAudio} title={labels.deleteAudio}>
                  <IconTrash />
                </button>
              </div>
              <div className="record-controls" style={{ marginTop: '0.75rem' }}>
                {!autoTranscribe && (
                  <button className="btn-primary"
                    onClick={() => runTranscription(recordedBlob)}
                    disabled={status === 'loading'}>
                    {status === 'loading' ? labels.transcribing : labels.transcribeBtn}
                  </button>
                )}
                {result && (
                  <button className="btn-secondary"
                    onClick={() => runTranscription(recordedBlob)}
                    disabled={status === 'loading'}>
                    ↻ {labels.redoTranscript}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Glossary visible DURING interpretation (Mariam feedback): terminology
            should stay available while performing, not only on the prep page. */}
        {glossary.length > 0 && (
          <details className="record-section" style={{ marginTop: '0.5rem' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem', color: 'var(--primary)' }}>
              📖 {labels.glossaryDuringRecording} ({glossary.length})
            </summary>
            <div className="table-responsive" style={{ marginTop: '0.6rem', maxHeight: 260, overflowY: 'auto' }}>
              <table className="glossary-table">
                <thead>
                  <tr>
                    <th>{labels.glossaryTermHeader || 'Term'}</th>
                    <th>{labels.glossaryArabicHeader || 'Arabic'}</th>
                    <th>{labels.glossaryFrenchHeader || 'French'}</th>
                    <th>{labels.glossaryEnglishHeader || 'English'}</th>
                  </tr>
                </thead>
                <tbody>
                  {glossary.map((item, i) => (
                    <tr key={i}>
                      <td><strong>{item.term}</strong></td>
                      <td className="arabic" dir="rtl">{glossaryArabicValue(item)}</td>
                      <td>{glossaryValue(item, ['french', 'French', 'fr', 'FR', 'french_term', 'term_fr', 'french_translation', 'translation_fr', 'français', 'francais'])}</td>
                      <td>{glossaryValue(item, ['english', 'English', 'en', 'EN', 'english_term', 'term_en', 'english_translation', 'translation_en'])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </div>

      {/* Results card */}
      {(status === 'loading' || result || error) && (
        <div className="card">
          {status === 'loading' && (
            <div className="transcribing-state">
              <div className="spinner" />
              <p>{labels.transcribing}</p>
            </div>
          )}
          {error && <div className="error-msg">{labels.errorPrefix}: {error}</div>}

          {result && (
            <div className="transcript-result">
              <div className="result-meta" style={{ marginBottom: '0.75rem' }}>
                {result.duration_seconds > 0 && <span>{result.duration_seconds}s</span>}
                <span>{(result.language_detected || '').toUpperCase()}</span>
              </div>

              {lowConfWords.length > 0 && (
                <div className="warning-msg" style={{ marginBottom: '0.75rem' }}>
                  ⚠️ {labels.lowConfidence}: {lowConfWords.slice(0, 10).map(w => w.word.trim()).join(', ')}
                </div>
              )}

              {/* Correct the recognizer's text BEFORE the evaluation runs, so an
                  ASR mistake is never scored as an interpreting mistake
                  (ETIB feedback 21 Aug 2026). */}
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center', marginBottom: '0.4rem' }}>
                <button className="btn-secondary btn-sm"
                  onClick={() => (editingTranscript ? saveTranscriptEdit() : startEditingTranscript())}>
                  {editingTranscript ? labels.editTranscriptDone : labels.editTranscriptBtn}
                </button>
                {result.edited_by_student && (
                  <span style={{ fontSize: '0.78rem', color: 'var(--sage)', fontWeight: 600 }}>
                    ✎ {labels.transcriptEditedBadge}
                  </span>
                )}
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--warm-gray)', marginBottom: '0.6rem', lineHeight: 1.55 }}>
                💡 {labels.editTranscriptHint}
              </p>
              {editingTranscript ? (
                <textarea
                  value={transcriptDraft}
                  onChange={e => setTranscriptDraft(e.target.value)}
                  rows={10}
                  className={isArabic ? 'arabic' : ''}
                  dir={isArabic ? 'rtl' : 'ltr'}
                  style={{ width: '100%', resize: 'vertical', padding: '0.7rem', borderRadius: 8,
                           border: '1px solid var(--border, #ddd)', fontFamily: 'inherit',
                           fontSize: '0.95rem', lineHeight: 1.7 }}
                />
              ) : (
                <TranscriptBubble
                  labels={labels}
                  text={result.full_text}
                  vocalizedText={result.vocalized_text}
                  isArabic={isArabic}
                  onRetranscribe={recordedBlob ? () => runTranscription(recordedBlob) : null}
                  onDelete={clearTranscript}
                />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Module D — Full Evaluation Report ────────────────────────────────────────

// Arabic final case-ending (i'rab) labels — Arabic term + a short romanised hint.
// The acoustic module reports raw keys (none/fatha/damma/kasra/tanwin_*).
const IRAAB_ENDINGS = {
  none:        ['بدون حركة', '∅'],
  fatha:       ['فتحة', 'a'],
  damma:       ['ضمة', 'u'],
  kasra:       ['كسرة', 'i'],
  tanwin_fath: ['تنوين فتح', 'an'],
  tanwin_damm: ['تنوين ضمّ', 'un'],
  tanwin_kasr: ['تنوين كسر', 'in'],
};
function IraabEnding({ code }) {
  const [ar, roman] = IRAAB_ENDINGS[code] || [code || '—', ''];
  return (
    <span className="arabic" dir="rtl">
      {ar}{roman ? <small style={{ opacity: 0.7 }}> ({roman})</small> : null}
    </span>
  );
}

function ScoreBar({ score, labels }) {
  const safeScore = Number.isFinite(Number(score)) ? Number(score) : 0;
  const pct = Math.round((safeScore / 10) * 100);
  score = safeScore;
  const color = score >= 7 ? '#2D5A4E' : score >= 5 ? '#1B3A6B' : '#8B3A2A';
  const circumference = 188.5;
  const offset = circumference - (circumference * pct / 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
      <div className="prog-ring" style={{ width: 64, height: 64 }}>
        <svg viewBox="0 0 70 70"><circle className="prog-ring-bg" cx="35" cy="35" r="30"/><circle className="prog-ring-fill" cx="35" cy="35" r="30" stroke={color} strokeDasharray={circumference} strokeDashoffset={offset}/></svg>
        <div className="prog-pct" style={{ fontSize: '0.85rem' }}>{score.toFixed(1)}</div>
      </div>
      <div>
        <div className="score-bar-wrap"><div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} /></div>
        <div style={{ fontSize: '0.74rem', color: 'var(--warm-gray)', marginTop: '0.3rem' }}>{pct}% — {score >= 7 ? (labels?.verdictGood || 'Good') : score >= 5 ? (labels?.verdictAcceptable || 'Acceptable') : (labels?.verdictNeedsWork || 'Needs work')}</div>
      </div>
    </div>
  );
}

function EvalSection({ title, icon, items, renderItem, emptyLabel }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="eval-section">
      <button className="eval-section-header" onClick={() => setOpen(o => !o)}>
        <span>{icon} {title}</span>
        <span className={`eval-badge ${items.length > 0 ? 'eval-badge-warn' : 'eval-badge-ok'}`}>
          {items.length}
        </span>
        <span className="eval-chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="eval-section-body">
          {items.length === 0
            ? <p className="eval-empty">✓ {emptyLabel}</p>
            : items.map((item, i) => <div key={i} className="eval-item">{renderItem(item)}</div>)
          }
        </div>
      )}
    </div>
  );
}

// ── Tashkeel / Pronunciation Panel (text comparison — no audio needed) ───────

function PronunciationPanel({ labels, lastTranscript, lastGeneratedScript }) {
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError]   = useState('');

  const language   = lastTranscript?.language || lastTranscript?.language_detected || 'ar';
  const sourceText = lastGeneratedScript?.script || '';
  const transcript = lastTranscript?.full_text  || '';

  if (language !== 'ar') return <div className="info-tip">ℹ️ {labels.needsArabic}</div>;
  if (!sourceText)       return <div className="info-tip">ℹ️ {labels.needsSource}</div>;

  async function runCompare() {
    setStatus('loading');
    setError('');
    setResult(null);
    try {
      const data = await tashkeelCompare(sourceText, transcript, language);
      setResult(data);
      setStatus('success');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  }

  const coverageColor = result
    ? (result.coverage_score >= 8 ? '#1a6b3c' : result.coverage_score >= 6 ? '#1B3A6B' : '#c0392b')
    : 'var(--color-text)';

  return (
    <div>
      <button className="btn-primary" onClick={runCompare} disabled={status === 'loading'}>
        {status === 'loading' ? labels.runningPronunciation : labels.runPronunciation}
      </button>

      {status === 'loading' && (
        <div className="transcribing-state" style={{ marginTop: '1rem' }}>
          <div className="spinner" /><p>{labels.runningPronunciation}</p>
        </div>
      )}
      {error && <div className="error-msg" style={{ marginTop: '0.75rem' }}>{labels.errorPrefix}: {error}</div>}

      {result && (
        <div style={{ marginTop: '1.5rem' }}>

          {/* Scores */}
          <div style={{ display: 'flex', gap: '2rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
            <div>
              <p className="report-label">{labels.coverageScoreLabel}</p>
              <div className="score-number" style={{ color: coverageColor }}>
                {result.coverage_score?.toFixed(1)}<span>/10</span>
              </div>
            </div>
            <div>
              <p className="report-label">{labels.overallScore}</p>
              <div className="score-number" style={{ color: coverageColor }}>
                {result.overall_score?.toFixed(1)}<span>/10</span>
              </div>
            </div>
          </div>

          {result.summary && (
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
              {result.summary}
            </p>
          )}

          {/* Missing information */}
          {(result.missing_content || []).length > 0 && (
            <div className="materials-section">
              <h4 style={{ color: '#c0392b', fontSize: '0.85rem', fontWeight: 700,
                textTransform: 'uppercase', marginBottom: '0.6rem' }}>
                📉 {labels.informationLoss} ({result.missing_content.length})
              </h4>
              {result.missing_content.map((item, i) => (
                <div key={i} className="eval-item">
                  <span className="eval-text">{item.content || item}</span>
                  {item.importance && (
                    <span className={`importance-badge importance-${item.importance}`}>
                      {item.importance}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Tashkeel errors */}
          {(result.tashkeel_errors || []).length > 0 && (
            <div className="materials-section">
              <h4 style={{ color: '#1B3A6B', fontSize: '0.85rem', fontWeight: 700,
                textTransform: 'uppercase', marginBottom: '0.6rem' }}>
                ⚠️ {labels.languageErrors} — إعراب / تشكيل ({result.tashkeel_errors.length})
              </h4>
              {result.tashkeel_errors.map((err, i) => (
                <div key={i} className="uncertain-word-card">
                  <div className="uncertain-word-header">
                    <span className="uncertain-word arabic">{err.word}</span>
                    {err.expected_form && (
                      <span className="detail-label" style={{ marginInlineStart: '0.5rem' }}>
                        → <span className="arabic" style={{ fontSize: '1.1rem', color: '#1a6b3c' }}>{err.expected_form}</span>
                      </span>
                    )}
                  </div>
                  {err.expected_case && (
                    <div className="uncertain-detail">
                      <span className="detail-label">{labels.grammaticalRole}:</span>
                      <span className="detail-value">{err.expected_case}</span>
                    </div>
                  )}
                  {err.likely_student_error && (
                    <div className="uncertain-detail">
                      <span className="detail-label">{labels.likelyError}:</span>
                      <span className="detail-value" style={{ color: '#c0392b' }}>{err.likely_student_error}</span>
                    </div>
                  )}
                  {err.explanation && <p className="uncertain-explanation">{err.explanation}</p>}
                </div>
              ))}
            </div>
          )}

          {/* Correct tashkeel */}
          {(result.tashkeel_correct || []).length > 0 && (
            <div className="materials-section">
              <h4 style={{ color: '#1a6b3c', fontSize: '0.85rem', fontWeight: 700,
                textTransform: 'uppercase', marginBottom: '0.6rem' }}>
                ✅ {labels.correctTashkeel} ({result.tashkeel_correct.length})
              </h4>
              <div className="key-terms-list">
                {result.tashkeel_correct.map((w, i) => (
                  <span key={i} className="key-term-badge arabic" style={{ fontSize: '1rem' }}
                    title={w.note}>{w.form || w.word}</span>
                ))}
              </div>
            </div>
          )}

          {(result.tashkeel_errors || []).length === 0 && (result.missing_content || []).length === 0 && (
            <div className="eval-item">✓ {labels.noIssues}</div>
          )}
        </div>
      )}
    </div>
  );
}

function ModuleD({ labels, lastTranscript, lastGeneratedScript, lastRecordingBlob, onEvaluationSaved }) {
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');

  const language = lastTranscript?.language || lastTranscript?.language_detected || 'ar';
  // Interpretation mode travels with the transcript (professor feedback: pauses
  // from normal Ear-Voice Span lag in simultaneous mode should not be penalized
  // as errors — the backend judges pauses against mode-aware criteria).
  const interpMode = lastTranscript?.mode || 'consecutive';

  async function handleEvaluate() {
    setStatus('loading');
    setError('');
    setReport(null);
    try {
      let data;
      if (lastRecordingBlob) {
        // Best path: re-transcribe locally for word timestamps + detect everything
        const audioFile = new File([lastRecordingBlob], 'recording.webm',
          { type: lastRecordingBlob.type });
        data = await evaluateWithAudio(
          audioFile,
          lastGeneratedScript?.script || '',
          language,
          lastGeneratedScript?.language || language,
          lastGeneratedScript?.domain || '',
          lastGeneratedScript?.glossary || [],   // student-reviewed glossary → terminology check
          interpMode,
          // Transcript the student corrected in Module C. The backend evaluates
          // this text instead of the raw ASR output, while pauses, hesitations
          // and speech rate stay measured from the real audio.
          lastTranscript?.edited_by_student ? (lastTranscript.full_text || '') : ''
        );
      } else {
        // Fallback: use stored Groq transcript (less accurate for hesitations)
        data = await generateFeedback({
          source_script:   lastGeneratedScript?.script || '',
          transcript_text: lastTranscript.full_text,
          transcript:      lastTranscript,
          language,
          glossary:        lastGeneratedScript?.glossary || [],
          mode:            interpMode
        });
      }
      // LLM step failed (HTTP 207) — no scores exist, so a partial report with
      // "0.0" would mislead the student. Show a clear error instead.
      if (data.llm_failed) {
        const errText = String(data.error || '').toLowerCase();
        const isQuota = /rate.?limit|quota|429|tokens per|tpd|tpm|capacity/.test(errText);
        throw new Error(isQuota
          ? (labels.llmQuotaError || 'The AI service is temporarily rate-limited — wait about a minute, then run the evaluation again. The shared limit resets on its own.')
          : (labels.llmDownError || 'The AI evaluation service did not respond — try again in a moment.'));
      }
      // Attach pronunciation report for the pronunciation panel
      if (data.pronunciation) {
        data._pronunciationForPanel = {
          overall_score: data.pronunciation.overall_score,
          uncertain_count: data.pronunciation.uncertain_count,
          flagged_words: data.pronunciation.flagged_words || [],
          summary: data.pronunciation.summary,
        };
      }
      setReport(data);
      setStatus('success');
      onEvaluationSaved?.();
    } catch (err) {
      setError(friendlyLlmError(err.message, labels));
      setStatus('error');
    }
  }

  if (!lastTranscript) {
    return (
      <div className="card coming-soon">
        <p className="eyebrow">{labels.moduleDTitle}</p>
        <p>{labels.noTranscriptYet}</p>
      </div>
    );
  }

  const algo = report?.algorithmic || {};
  const fluency = report?.fluency || null;
  // ETIB decision (21 Aug 2026): the automatic meaning/content verdict is not
  // shown at all — the backend drops it (EVALUATION_REPORT_MEANING=false) and
  // the report must not render an empty shell for it either. Set the env var to
  // true to bring the meaning sections back as informational feedback.
  const showMeaning = report?.report_meaning !== false;
  const isAr = (lastTranscript.language || lastTranscript.language_detected) === 'ar';
  const llmNumberErrors = (report?.number_accuracy || [])
    .filter(item => item && item.correct === false)
    .map(item => ({ ...item }));
  const displayNumberErrors = llmNumberErrors.length > 0 ? llmNumberErrors : (algo.number_errors || []);

  return (
    <div>
      <div className="card">
        <h2>{labels.moduleDTitle}</h2>
        <div className="info-tip" style={{ marginBottom: '1rem' }}>
          📄 {labels.transcriptReady} ({lastTranscript.duration_seconds ?? 0}s · {(lastTranscript.language_detected || '').toUpperCase()})
          {lastGeneratedScript && <> · {labels.source}: {lastGeneratedScript.domain}</>}
          {lastTranscript.edited_by_student && <> · ✎ {labels.transcriptEditedBadge}</>}
        </div>
        <button className="btn-primary" onClick={handleEvaluate} disabled={status === 'loading'}>
          {status === 'loading' ? labels.runningEvaluation : labels.runEvaluation}
        </button>
        {status === 'loading' && (
          <div className="transcribing-state" style={{ marginTop: '1rem' }}>
            <div className="spinner" /><p>{labels.runningEvaluation}</p>
          </div>
        )}
        {error && <div className="error-msg" style={{ marginTop: '1rem' }}>{labels.errorPrefix}: {error}</div>}
      </div>

      {report && (
        <div className="report-card">

          {/* LLM partial failure warning */}
          {report.llm_failed && (
            <div className="warning-msg" style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              ⚠️ {labels.llmFailedWarning}
            </div>
          )}

          {/* Score + Summary */}
          <div className="card report-header">
            <div className="report-score-row">
              <div>
                <p className="report-label">{labels.overallScore}</p>
                <ScoreBar score={report.overall_score || 0} labels={labels} />
              </div>
              {report.fluency_score !== undefined && (
                <div>
                  <p className="report-label">{labels.fluencyScore}</p>
                  <ScoreBar score={report.fluency_score || 0} labels={labels} />
                </div>
              )}
              {(report.strengths || []).length > 0 && (
                <div className="report-strengths">
                  <p className="report-label">✅ {labels.strengths}</p>
                  <ul>{(report.strengths || []).map((s, i) => <li key={i} dir="auto">{s}</li>)}</ul>
                </div>
              )}
            </div>
            {report.summary && (
              <div className="report-summary">
                <p className="report-label">📋 {labels.evalSummary}</p>
                <p dir="auto" style={{ textAlign: 'start' }}>{report.summary}</p>
              </div>
            )}
            {report.score_on_meaning === false && showMeaning && (
              <div className="info-tip" style={{ marginTop: '0.75rem', fontSize: '0.8rem' }}>
                ℹ️ {labels.scoreScopeNote}
              </div>
            )}
            {report.transcript_edited && (
              <div className="info-tip" style={{ marginTop: '0.75rem', fontSize: '0.8rem' }}>
                ✎ {labels.transcriptEditedBadge}
              </div>
            )}
          </div>

          {/* Algorithmic detections */}
          <div className="card">
            <h3 className="report-section-title">🔍 {labels.algorithmicTitle}</h3>
            <div className="algo-grid">
              {[
                { label: labels.longSilences,  icon: '🔇', items: algo.long_silences || [] },
                { label: labels.repetitions,   icon: '🔁', items: algo.repetitions || [] },
                { label: labels.hesitations,   icon: '🗣️', items: algo.hesitation_words || [] },
                { label: labels.numberErrors,  icon: '🔢', items: displayNumberErrors },
              ].map(({ label, icon, items }) => (
                <div key={label} className={`algo-card ${items.length > 0 ? 'algo-card-warn' : ''}`}>
                  <span className="algo-icon">{icon}</span>
                  <span className="algo-count">{items.length}</span>
                  <span className="algo-label">{label}</span>
                </div>
              ))}
            </div>
            {/* Detail lists */}
            {(algo.long_silences || []).length > 0 && (
              <div className="algo-detail-block">
                <p className="algo-detail-title">🔇 {labels.longSilences}</p>
                {algo.long_silences.map((s, i) => {
                  const isPrep = String(s.type || '').includes('leading') || Number(s.at_seconds) === 0;
                  return (
                    <div key={i} className="algo-detail-row">
                      <span className={`algo-detail-badge ${isPrep ? 'algo-badge-gold' : 'algo-badge-warn'}`}>{s.duration_seconds}s</span>
                      <span className="algo-detail-text">
                        {isPrep
                          ? <>⏱ {labels.prepTimeLabel}</>
                          : <>at {s.at_seconds}s{s.after_text ? ` after: "${s.after_text}"` : ''}</>}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
            {(algo.repetitions || []).length > 0 && (
              <div className="algo-detail-block">
                <p className="algo-detail-title">🔁 {labels.repetitions}</p>
                {algo.repetitions.map((r, i) => (
                  <div key={i} className="algo-detail-row">
                    <span className="algo-detail-badge algo-badge-gold">"{r.word}"</span>
                    <span className="algo-detail-text">
                      {r.at_seconds !== undefined
                        ? `at ${r.at_seconds}s${r.second_occurrence !== undefined ? ` · again at ${r.second_occurrence}s` : ''}`
                        : `near word ${r.position ?? i}`}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {(algo.hesitation_words || []).length > 0 && (
              <div className="algo-detail-block">
                <p className="algo-detail-title">🗣️ {labels.hesitations}</p>
                <div className="algo-chips">
                  {algo.hesitation_words.map((h, i) => {
                    const raw = h?.word || h;
                    const display = raw === '[filled pause]' ? (labels.filledPause || 'filled pause')
                      : (raw === '[cut word]' || raw === '[false start]') ? (labels.cutWord || 'cut word')
                      : raw;
                    return (
                      <span key={i} className="algo-chip">
                        "{display}" <small>{h?.at_seconds !== undefined ? `${h.at_seconds}s` : ''}</small>
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
            {/* Raw number chips only when the AI Numbers & dates section is absent —
                two number blocks side by side confused evaluators (16 July feedback) */}
            {displayNumberErrors.length > 0 && !(report.number_accuracy || []).length && (
              <div className="algo-detail-block">
                <p className="algo-detail-title">🔢 {labels.numberErrors}</p>
                <div className="algo-chips">
                  {displayNumberErrors.map((n, i) => (
                    <span key={i} className="algo-chip algo-chip-err">
                      {typeof n === 'string'
                        ? n
                        : `${n.source_value || n.expected || ''} → ${n.student_said || n.heard || '—'}`}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Pause & flow card REMOVED (user request 21 Aug 2026): its LLM
              comment kept drifting into meaning/coverage territory ("toute la
              suite du discours non restituée"), which is out of scope now.
              The factual silence list stays in the algorithmic block above. */}

          {/* Repetition analysis */}
          {report.repetition_analysis?.comment && (
            <div className="card">
              <h3 className="report-section-title">🔁 {labels.repetitionAnalysisTitle}</h3>
              <p className={isAr ? 'arabic' : ''} style={{ fontSize: '0.85rem', color: 'var(--ink)' }}>{report.repetition_analysis.comment}</p>
            </div>
          )}

          {/* Number & date accuracy */}
          {(report.number_accuracy || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">🔢 {labels.numberAccuracyTitle}</h3>
              {report.number_accuracy.map((n, i) => (
                <div key={i} className="eval-item" style={{ borderInlineStart: `3px solid ${n.correct ? 'var(--sage)' : 'var(--sienna)'}`, paddingInlineStart: '0.75rem', marginBottom: '0.6rem' }}>
                  <div style={{ fontSize: '0.84rem' }}>
                    <strong>{n.source_value}</strong>
                    <span style={{ color: 'var(--warm-gray)' }}> — {labels.expectedLabel}: </span>
                    <strong style={{ color: 'var(--sage)' }} className={isAr ? 'arabic' : ''}>{n.expected_in_target}</strong>
                    <span style={{ color: 'var(--warm-gray)' }}> · {labels.studentSaidShort}: </span>
                    <strong style={{ color: n.correct ? 'var(--sage)' : 'var(--sienna)' }} className={isAr ? 'arabic' : ''}>{n.student_said || '—'}</strong>
                    <span className={`importance-badge importance-${n.correct ? 'low' : 'high'}`} style={{ marginInlineStart: '0.5rem' }}>
                      {n.correct ? labels.numCorrect : labels.numIncorrect}
                    </span>
                  </div>
                  {n.note && <div className="eval-explanation" dir="auto">{n.note}</div>}
                </div>
              ))}
            </div>
          )}

          {/* Meaning check (informational) — clear mistranslations shown to the
              student with WHERE + the CORRECT form (Kevin, 12 Aug: a wrong-meaning
              word wasn't being surfaced). Re-added after being removed 16 July;
              does NOT affect the score (score_on_meaning stays off), and valid
              paraphrases are never listed. */}
          {showMeaning && (report.translation_errors || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">🧭 {labels.meaningCheckTitle}</h3>
              <p className="info-tip" style={{ fontSize: '0.8rem', marginBottom: '0.6rem' }}>ℹ️ {labels.meaningCheckNote}</p>
              {report.translation_errors.map((t, i) => (
                <div key={i} className="eval-item" style={{ borderInlineStart: '3px solid var(--sienna)', paddingInlineStart: '0.75rem', marginBottom: '0.6rem' }}>
                  <div style={{ fontSize: '0.84rem' }}>
                    <span style={{ color: 'var(--warm-gray)' }}>{labels.sourceText}: </span>
                    <strong dir="auto">{t.source_text}</strong>
                    <span style={{ color: 'var(--warm-gray)' }}> · {labels.studentSaidShort}: </span>
                    <strong style={{ color: 'var(--sienna)' }} className={isAr ? 'arabic' : ''} dir="auto">{t.student_said || '—'}</strong>
                    {t.correct_translation && (<>
                      <span style={{ color: 'var(--warm-gray)' }}> · {labels.expectedLabel}: </span>
                      <strong style={{ color: 'var(--sage)' }} className={isAr ? 'arabic' : ''} dir="auto">{t.correct_translation}</strong>
                    </>)}
                  </div>
                  {t.explanation && <div className="eval-explanation" dir="auto">{t.explanation}</div>}
                </div>
              ))}
            </div>
          )}

          {/* Arabic case endings (إعراب) — acoustic detection from the
              new_arabic_solution module. Reports what the student ACTUALLY
              pronounced from the audio; the transcript is left verbatim and
              wrong endings are never "corrected". */}
          {report.arabic_module?.pronunciation?.available && (() => {
            const pron = report.arabic_module.pronunciation;
            const sum = pron.summary || {};
            const acc = sum.accuracy != null ? Math.round(sum.accuracy * 100) : null;
            const findings = pron.findings || [];
            return (
              <div className="card">
                <h3 className="report-section-title">📐 {labels.iraabTitle}</h3>
                <p className="info-tip" style={{ fontSize: '0.8rem', marginBottom: '0.75rem' }}>
                  ℹ️ {labels.iraabAcousticNote}
                </p>
                {/* Ali's vowelized (tashkeel/tanween) hypothesis — shown only when
                    it actually carries harakat (i.e. once his diacritizer is on). */}
                {/[ً-ْ]/.test(report.arabic_module.transcript?.diacritized || '') && (
                  <div className="eval-item" style={{ marginBottom: '0.75rem', paddingInlineStart: '0.75rem', borderInlineStart: '3px solid var(--gold)' }}>
                    <span style={{ color: 'var(--warm-gray)', fontSize: '0.78rem' }}>{labels.iraabDiacritized}: </span>
                    <span className="arabic" dir="rtl" style={{ fontSize: '1rem' }}>{report.arabic_module.transcript.diacritized}</span>
                  </div>
                )}
                <div className="algo-grid" style={{ marginBottom: '0.75rem' }}>
                  <div className="algo-card">
                    <span className="algo-count" style={{ color: 'var(--sage)' }}>{sum.correct ?? 0}</span>
                    <span className="algo-label">{labels.iraabCorrect}</span>
                  </div>
                  <div className={`algo-card ${(sum.incorrect ?? 0) > 0 ? 'algo-card-warn' : ''}`}>
                    <span className="algo-count" style={{ color: 'var(--sienna)' }}>{sum.incorrect ?? 0}</span>
                    <span className="algo-label">{labels.iraabIncorrect}</span>
                  </div>
                  <div className="algo-card">
                    <span className="algo-count" style={{ color: 'var(--gold)' }}>{sum.uncertain ?? 0}</span>
                    <span className="algo-label">{labels.iraabUncertain}</span>
                  </div>
                  {acc != null && (
                    <div className="algo-card">
                      <span className="algo-count">{acc}%</span>
                      <span className="algo-label">{labels.iraabAccuracy}</span>
                    </div>
                  )}
                </div>
                {findings.map((f, i) => {
                  const color = f.status === 'correct' ? 'var(--sage)'
                    : f.status === 'incorrect' ? 'var(--sienna)' : 'var(--gold)';
                  const badge = f.status === 'correct' ? labels.iraabCorrect
                    : f.status === 'incorrect' ? labels.iraabIncorrect : labels.iraabUncertain;
                  const importance = f.status === 'incorrect' ? 'high' : 'low';
                  return (
                    <div key={i} className="eval-item" style={{ borderInlineStart: `3px solid ${color}`, paddingInlineStart: '0.75rem', marginBottom: '0.6rem' }}>
                      <div style={{ fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                        <strong className="arabic" dir="rtl" style={{ fontSize: '1.05rem' }}>{f.word}</strong>
                        <span className={`importance-badge importance-${importance}`}>{badge}</span>
                        {f.detection_confidence != null && (
                          <small style={{ color: 'var(--warm-gray)' }}>
                            {labels.pronunciationScore}: {Math.round(f.detection_confidence * 100)}%
                          </small>
                        )}
                      </div>
                      <div style={{ fontSize: '0.84rem', marginTop: '0.3rem' }}>
                        <span style={{ color: 'var(--warm-gray)' }}>{labels.expectedLabel}: </span>
                        <IraabEnding code={f.expected_ending} />
                        <span style={{ color: 'var(--warm-gray)', margin: '0 0.4rem' }}>→</span>
                        <span style={{ color: 'var(--warm-gray)' }}>{labels.iraabDetected}: </span>
                        <span style={{ color }}><IraabEnding code={f.detected_ending} /></span>
                      </div>
                      {f.explanation && <div className="eval-explanation" dir="auto">{f.explanation}</div>}
                    </div>
                  );
                })}
              </div>
            );
          })()}

          {/* Pronunciation assessment (LLM commentary, all languages) */}
          {false && report.pronunciation_assessment?.comment && (
            <div className="card">
              <h3 className="report-section-title">🗣️ {labels.pronunciationAssessmentTitle}</h3>
              <p className={isAr ? 'arabic' : ''} style={{ fontSize: '0.85rem', color: 'var(--ink)', marginBottom: '0.6rem' }}>{report.pronunciation_assessment.comment}</p>
              {(report.pronunciation_assessment.issues || []).map((item, i) => (
                <div key={i} className="eval-item" style={{ borderInlineStart: '3px solid var(--gold)', paddingInlineStart: '0.75rem', marginBottom: '0.5rem' }}>
                  <strong className={isAr ? 'arabic' : ''}>"{item.word}"</strong>
                  {item.issue && <div className="eval-explanation" dir="auto">{item.issue}</div>}
                  {item.correction && <div className="eval-correction">✓ {labels.correction}: <strong className={isAr ? 'arabic' : ''}>{item.correction}</strong></div>}
                </div>
              ))}
            </div>
          )}

          {/* Audio-based fluency */}
          {fluency && (
            <div className="card">
              <h3 className="report-section-title">{labels.audioFluencyTitle}</h3>
              <div className="report-score-row">
                <div>
                  <p className="report-label">{labels.fluencyScore}</p>
                  <ScoreBar score={fluency.fluency_score || 0} labels={labels} />
                </div>
                <div className="report-summary">
                  <p>{fluency.summary}</p>
                  <p style={{ fontSize: '0.82rem', color: 'var(--warm-gray)', marginTop: '0.45rem' }}>
                    {labels.fluencyExplain}
                  </p>
                  {fluency.preparation_seconds > 0 && (
                    <p style={{ fontSize: '0.82rem', color: 'var(--sage)', marginTop: '0.35rem' }}>
                      ⏱ {fluency.preparation_seconds}s {labels.prepTimeLabel}
                    </p>
                  )}
                </div>
              </div>
              <div className="algo-grid" style={{ marginTop: '1rem' }}>
                {[
                  { label: labels.statSpeechRate, value: `${fluency.speech_rate_wpm || 0} wpm` },
                  { label: labels.statLongPauses, value: fluency.long_pause_count || 0 },
                  { label: labels.statSilenceRatio, value: `${Math.round((fluency.silence_ratio || 0) * 100)}%` },
                  { label: labels.statWordConfidence, value: `${Math.round((fluency.average_word_confidence || 0) * 100)}%` },
                ].map(({ label, value }) => (
                  <div key={label} className="algo-card">
                    <span className="algo-count">{value}</span>
                    <span className="algo-label">{label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Coverage score */}
          {showMeaning && report.coverage_score != null && (
            <div className="card">
              <h3 className="report-section-title">📊 {labels.coverageTitle}</h3>
              <ScoreBar score={report.coverage_score ?? 0} labels={labels} />
              <p style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--warm-gray)' }}>
                {labels.coverageHint}
              </p>
            </div>
          )}

          {/* Translation errors section REMOVED (professor feedback 16 July):
              word-level source-vs-target comparison is machine-translation
              thinking — an interpreter transmits meaning, not words. Meaning-
              level feedback lives in missing content, terminology (vs the
              approved glossary), numbers, and proper nouns. */}

          {/* Proper nouns — names of people, organizations, places */}
          {(report.proper_nouns || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">🏛️ {labels.pnTitle}</h3>
              {report.proper_nouns.map((pn, i) => {
                const status = String(pn.status || '').toLowerCase();
                const color = status === 'correct' ? 'var(--sage)' : status === 'missing' ? 'var(--warm-gray)' : 'var(--sienna)';
                const statusLabel = status === 'correct' ? labels.pnCorrect : status === 'missing' ? labels.pnMissing : labels.pnDistorted;
                return (
                  <div key={i} className="eval-item" style={{ borderInlineStart: `3px solid ${color}`, paddingInlineStart: '0.75rem', marginBottom: '0.6rem' }}>
                    <div style={{ fontSize: '0.84rem' }}>
                      <strong>{pn.source_name}</strong>
                      {pn.student_said && status !== 'missing' && (
                        <>
                          <span style={{ color: 'var(--warm-gray)' }}> · {labels.studentSaidShort}: </span>
                          <strong style={{ color }} className={isAr ? 'arabic' : ''}>{pn.student_said}</strong>
                        </>
                      )}
                      <span className={`importance-badge importance-${status === 'correct' ? 'low' : 'high'}`} style={{ marginInlineStart: '0.5rem' }}>
                        {statusLabel}
                      </span>
                    </div>
                    {pn.note && <div className="eval-explanation" dir="auto">{pn.note}</div>}
                  </div>
                );
              })}
            </div>
          )}

          {/* Missing content */}
          {showMeaning && (report.missing_content || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">📉 {labels.missingContent}</h3>
              {(report.missing_content || []).map((item, i) => (
                <div key={i} className="eval-item" style={{ marginBottom: '0.5rem' }}>
                  <span className="eval-text">{item.content}</span>
                  {item.importance && <span className={`importance-badge importance-${item.importance}`}>{item.importance}</span>}
                </div>
              ))}
            </div>
          )}


          {/* Auto-corrections (self-corrections the student made mid-speech) */}
          {(report.auto_corrections || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">✏️ {labels.autoCorrections}</h3>
              {report.auto_corrections.map((item, i) => (
                <div key={i} className="eval-item" style={{ borderInlineStart: '3px solid var(--sage)', paddingInlineStart: '0.75rem', marginBottom: '0.5rem' }}>
                  <div className={`eval-text ${isAr ? 'arabic' : ''}`}>"{item.text}"</div>
                </div>
              ))}
            </div>
          )}

          {/* False starts (incomplete phrases abandoned mid-sentence) */}
          {(report.false_starts || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">↩️ {labels.falseStarts}</h3>
              {report.false_starts.map((item, i) => (
                <div key={i} className="eval-item" style={{ borderInlineStart: '3px solid var(--sienna)', paddingInlineStart: '0.75rem', marginBottom: '0.5rem' }}>
                  <div className={`eval-text ${isAr ? 'arabic' : ''}`}>"{item.text}"</div>
                </div>
              ))}
            </div>
          )}

          {/* Lapsus linguae (slips of the tongue) */}
          {(report.lapsus_linguae || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">👅 {labels.lapsusLinguae}</h3>
              {report.lapsus_linguae.map((item, i) => (
                <div key={i} className="eval-item" style={{ borderInlineStart: '3px solid var(--gold)', paddingInlineStart: '0.75rem', marginBottom: '0.5rem' }}>
                  <div className={`eval-text ${isAr ? 'arabic' : ''}`}>"{item.text}"</div>
                  {item.likely_intended && <div className="eval-correction">→ {labels.correction}: <strong className={isAr ? 'arabic' : ''}>{item.likely_intended}</strong></div>}
                </div>
              ))}
            </div>
          )}

          {/* Recommendations */}
          {(report.recommendations || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">💡 {labels.recommendations}</h3>
              <ul className="recommendations-list">
                {report.recommendations.map((r, i) => <li key={i} dir="auto">{r}</li>)}
              </ul>
            </div>
          )}

        </div>
      )}

    </div>
  );
}

function ComingSoon({ title, body, labels, detail }) {
  return (
    <div className="card coming-soon">
      <p className="eyebrow">{labels.comingSoon}</p>
      <h2>{title}</h2>
      <p>{body}</p>
      {detail && <p className="module-context">A generated speech is ready for this module once backend endpoints are connected.</p>}
    </div>
  );
}
