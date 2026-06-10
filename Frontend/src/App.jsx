import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  generateSpeech,
  generateSpeechFromDocument,
  loginUser,
  logoutUser,
  retrieveDocumentContext,
  signupUser,
  textToSpeech,
  generateMaterials,
  downloadGlossary,
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
} from './api.js';

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
    instructor: 'Instructor',
    coordinator: 'Coordinator',
    signOut: 'Sign out',
    workspaceKicker: 'Training modules',
    workspaceTitle: 'Generate speeches, prepare materials, record interpretations, and review feedback.',
    navA: 'Generate speech',
    navB: 'Audio & materials',
    navC: 'Record & transcribe',
    navD: 'Evaluation',
    moduleATitle: 'Generate training speech',
    groundedSourceLabel: 'Grounded in real UN document:',
    ungroundedNote: 'No matching UN document was found — this speech was generated without source grounding.',
    language: 'Speech language',
    targetLanguage: 'Target language',
    topic: 'Topic',
    domain: 'Domain',
    wordCount: 'Word count',
    difficulty: 'Difficulty',
    structure: 'Discourse structure',
    mode: 'Interpretation mode',
    numbers: 'Number density',
    hesitations: 'Simulate hesitations',
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
    retrieveContext: 'Preview retrieved context',
    retrievedContext: 'Retrieved context',
    submit: 'Generate speech',
    generating: 'Generating - please wait...',
    wordsUnit: 'words',
    moduleBTitle: 'Audio & Pedagogical Materials',
    moduleBBody: 'Will produce: audio playback, key terms, glossary, MCQ, and flashcards.',
    moduleCTitle: 'Record & Transcribe',
    moduleCBody: 'Will allow: in-browser recording, ASR transcription, timestamped transcript, and playback.',
    moduleDTitle: 'Performance Evaluation',
    moduleDBody: 'Will detect: hesitations, omissions, number errors, terminology issues, and recurrent weaknesses.',
    comingSoon: 'Coming soon',
    errorPrefix: 'Error',
    noSpeechYet: 'Generate a speech in Module A first, then come back here.',
    voiceAccent: 'Voice accent',
    speechRate: 'Speech rate',
    generateAudio: 'Generate audio',
    generatingAudio: 'Generating audio...',
    materialsTitle: 'Pedagogical materials',
    generateMaterials: 'Generate materials',
    generatingMaterials: 'Generating materials...',
    keyTerms: 'Key terms',
    summaryTitle: 'Thematic summary',
    mcqTitle: 'Comprehension questions (MCQ)',
    glossaryTitle: 'Trilingual glossary (AR – FR – EN)',
    downloadGlossary: 'Download glossary (DOCX)',
    uploadAudio: 'Upload your interpretation (MP3, WAV, M4A)',
    transcribeBtn: 'Transcribe',
    transcribing: 'Transcribing — may take 1–2 min for Arabic...',
    transcriptLabel: 'Transcript',
    lowConfidence: 'Low-confidence words (possible errors)',
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
    sourceSaid: 'Source said',
    studentSaid: 'Student said',
    correctTranslation: 'Correct',
    missingContent: 'Missing content',
    pronunciationMismatches: 'Word-level mismatches vs source',
    pronunciationTitle: 'Pronunciation Assessment (إعراب)',
    runPronunciation: 'Run pronunciation check',
    runningPronunciation: 'Aligning audio against source — may take 1 min...',
    pronunciationScore: 'Pronunciation confidence',
    wordView: 'Word-by-word view',
    legend: 'Legend',
    legendGood: '≥80% confident',
    legendWarn: '65–80%',
    legendPoor: '<65% — likely error',
    uncertainWords: 'Uncertain words — likely pronunciation errors',
    expectedForm: 'Expected',
    likelyError: 'Likely error',
    grammaticalRole: 'Grammatical role',
    noUncertain: 'All words pronounced with high confidence',
    whisperxBadge: 'WhisperX aligned',
    fallbackBadge: 'Whisper scores',
    needsArabic: 'Pronunciation check is available for Arabic interpretations only.',
    needsSource: 'Generate a speech in Module A first so the source text is available.',
    sourceAudio: 'Source speech audio',
    yourInterpretation: 'Your interpretation',
    recordBtn: '🎙 Start recording',
    stopBtn: '⏹ Stop recording',
    recordingLive: 'Recording',
    autoTranscribe: 'Transcribe automatically after recording',
    orUpload: 'Or upload an audio file',
    copyText: 'Copy',
    copied: 'Copied!',
    topicPlaceholder: 'Enter a topic or paste text to generate a speech…',
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
    diffBeginner: 'Beginner',
    diffIntermediate: 'Intermediate',
    diffAdvanced: 'Advanced',
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
    instructor: 'مدرب',
    coordinator: 'منسق',
    signOut: 'تسجيل الخروج',
    workspaceKicker: 'وحدات التدريب',
    workspaceTitle: 'توليد الخطابات، إعداد المواد، تسجيل الأداء، ومراجعة التغذية الراجعة.',
    navA: 'توليد الخطاب',
    navB: 'الصوت والمواد',
    navC: 'التسجيل والتفريغ',
    navD: 'التقييم',
    moduleATitle: 'توليد خطاب تدريبي',
    groundedSourceLabel: 'مستند إلى وثيقة أممية حقيقية:',
    ungroundedNote: 'لم يتم العثور على وثيقة أممية مطابقة — تم إنشاء هذا الخطاب دون الاستناد إلى مصدر.',
    language: 'لغة الخطاب',
    targetLanguage: 'لغة الترجمة الهدف',
    topic: 'الموضوع',
    domain: 'المجال',
    wordCount: 'عدد الكلمات',
    difficulty: 'مستوى الصعوبة',
    structure: 'بنية الخطاب',
    mode: 'نوع الترجمة الفورية',
    numbers: 'كثافة الأرقام',
    hesitations: 'محاكاة التردد',
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
    voiceAccent: 'نبرة الصوت',
    speechRate: 'سرعة الإلقاء',
    generateAudio: 'توليد الصوت',
    generatingAudio: 'جارٍ توليد الصوت...',
    materialsTitle: 'المواد التعليمية',
    generateMaterials: 'توليد المواد',
    generatingMaterials: 'جارٍ توليد المواد...',
    keyTerms: 'المصطلحات الأساسية',
    summaryTitle: 'الملخص الموضوعي',
    mcqTitle: 'أسئلة الفهم (اختيار متعدد)',
    glossaryTitle: 'المسرد الثلاثي (عربي – فرنسي – إنجليزي)',
    downloadGlossary: 'تنزيل المسرد (DOCX)',
    uploadAudio: 'ارفع تسجيل ترجمتك (MP3, WAV, M4A)',
    transcribeBtn: 'فرّغ الصوت',
    transcribing: 'جارٍ التفريغ — قد يستغرق دقيقتين للعربية...',
    transcriptLabel: 'النص المفرَّغ',
    lowConfidence: 'كلمات غير مؤكدة (أخطاء محتملة)',
    arabicTip: 'للحصول على أفضل النتائج: تحدث بالعربية الفصحى في غرفة هادئة.',
    noTranscriptYet: 'سجّل ترجمتك أولاً في "التسجيل والتفريغ"، ثم عد إلى هنا.',
    runEvaluation: 'تشغيل التقييم الكامل',
    runningEvaluation: 'جارٍ تحليل الأداء...',
    overallScore: 'التقييم الإجمالي',
    evalSummary: 'التقييم العام',
    strengths: 'نقاط القوة',
    recommendations: 'التوصيات',
    algorithmicTitle: 'الرصد التلقائي',
    longSilences: 'صمت طويل',
    repetitions: 'تكرارات',
    hesitations: 'تردد',
    numberErrors: 'أخطاء في الأرقام',
    llmTitle: 'التحليل اللغوي بالذكاء الاصطناعي',
    languageErrors: 'أخطاء لغوية',
    autoCorrections: 'تصحيحات ذاتية',
    falseStarts: 'فروع زائفة',
    lapsusLinguae: 'زلّات لسان',
    terminologyProblems: 'مشاكل مصطلحية',
    informationLoss: 'فقدان معلومات',
    noIssues: 'لا توجد مشكلات',
    correction: 'التصحيح',
    source: 'المصدر',
    used: 'ما قيل',
    correct: 'المصطلح الصحيح',
    coverageTitle: 'تغطية المحتوى',
    coverageHint: 'مقدار ما تم نقله من الخطاب المصدر',
    translationErrors: 'أخطاء الترجمة',
    sourceSaid: 'ما قاله المصدر',
    studentSaid: 'ما قاله الطالب',
    correctTranslation: 'الترجمة الصحيحة',
    missingContent: 'محتوى مفقود',
    pronunciationMismatches: 'تباينات على مستوى الكلمات',
    pronunciationTitle: 'تقييم النطق والإعراب',
    runPronunciation: 'تشغيل فحص النطق',
    runningPronunciation: 'جارٍ المحاذاة الصوتية — قد تستغرق دقيقة...',
    pronunciationScore: 'ثقة النطق',
    wordView: 'عرض كلمة بكلمة',
    legend: 'مفتاح الألوان',
    legendGood: '٪80 أو أكثر — صحيح',
    legendWarn: '٪65–80',
    legendPoor: 'أقل من ٪65 — خطأ محتمل',
    uncertainWords: 'كلمات غير مؤكدة — أخطاء نطق محتملة',
    expectedForm: 'الشكل الصحيح',
    likelyError: 'الخطأ المحتمل',
    grammaticalRole: 'الدور النحوي',
    noUncertain: 'جميع الكلمات نُطقت بثقة عالية',
    whisperxBadge: 'محاذاة WhisperX',
    fallbackBadge: 'نتائج Whisper',
    needsArabic: 'فحص النطق متاح للترجمات العربية فقط.',
    needsSource: 'ولّد خطاباً في الوحدة أ أولاً لتوفير النص المرجعي.',
    sourceAudio: 'الصوت المرجعي للخطاب',
    yourInterpretation: 'ترجمتك الفورية',
    recordBtn: '🎙 بدء التسجيل',
    stopBtn: '⏹ إيقاف التسجيل',
    recordingLive: 'جارٍ التسجيل',
    autoTranscribe: 'تفريغ تلقائي بعد التسجيل',
    orUpload: 'أو ارفع ملف صوتي',
    copyText: 'نسخ',
    copied: 'تم النسخ!',
    topicPlaceholder: 'أدخل موضوعاً أو الصق نصاً لتوليد خطاب...',
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
    diffBeginner: 'مبتدئ',
    diffIntermediate: 'متوسط',
    diffAdvanced: 'متقدم',
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
    instructor: 'Formateur',
    coordinator: 'Coordinateur',
    signOut: 'Se déconnecter',
    workspaceKicker: 'Modules d’entraînement',
    workspaceTitle: 'Générez des discours, préparez les supports, enregistrez les interprétations et consultez les retours.',
    navA: 'Générer un discours',
    navB: 'Audio et supports',
    navC: 'Enregistrer et transcrire',
    navD: 'Évaluation',
    moduleATitle: 'Générer un discours d’entraînement',
    groundedSourceLabel: 'Basé sur un document réel de l’ONU :',
    ungroundedNote: 'Aucun document de l’ONU correspondant n’a été trouvé — ce discours a été généré sans source.',
    language: 'Langue du discours',
    targetLanguage: 'Langue cible',
    domain: 'Domaine',
    wordCount: 'Nombre de mots',
    difficulty: 'Difficulté',
    structure: 'Structure du discours',
    mode: 'Mode d’interprétation',
    numbers: 'Densité des chiffres',
    hesitations: 'Simuler les hésitations',
    submit: 'Générer le discours',
    generating: 'Génération en cours...',
    wordsUnit: 'mots',
    moduleBTitle: 'Audio et supports pédagogiques',
    moduleBBody: 'Produira : lecture audio, termes clés, glossaire, QCM et flashcards.',
    moduleCTitle: 'Enregistrer et transcrire',
    moduleCBody: 'Permettra : enregistrement dans le navigateur, transcription ASR, horodatage et lecture.',
    moduleDTitle: 'Évaluation des performances',
    moduleDBody: 'Détectera : hésitations, omissions, erreurs de chiffres, problèmes terminologiques et faiblesses récurrentes.',
    comingSoon: 'Bientôt disponible',
    errorPrefix: 'Erreur',
    noSpeechYet: 'Générez d\'abord un discours dans le Module A, puis revenez ici.',
    voiceAccent: 'Accent vocal',
    speechRate: 'Débit de parole',
    generateAudio: 'Générer l\'audio',
    generatingAudio: 'Génération audio...',
    materialsTitle: 'Supports pédagogiques',
    generateMaterials: 'Générer les supports',
    generatingMaterials: 'Génération des supports...',
    keyTerms: 'Termes clés',
    summaryTitle: 'Résumé thématique',
    mcqTitle: 'Questions de compréhension (QCM)',
    glossaryTitle: 'Glossaire trilingue (AR – FR – EN)',
    downloadGlossary: 'Télécharger le glossaire (DOCX)',
    uploadAudio: 'Déposez votre interprétation (MP3, WAV, M4A)',
    transcribeBtn: 'Transcrire',
    transcribing: 'Transcription en cours — 1–2 min pour l\'arabe...',
    transcriptLabel: 'Transcription',
    lowConfidence: 'Mots peu fiables (erreurs possibles)',
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
    sourceSaid: 'Source a dit',
    studentSaid: "L'étudiant a dit",
    correctTranslation: 'Traduction correcte',
    missingContent: 'Contenu manquant',
    pronunciationMismatches: 'Divergences mot par mot',
    pronunciationTitle: 'Évaluation de la prononciation (إعراب)',
    runPronunciation: 'Lancer le contrôle de prononciation',
    runningPronunciation: 'Alignement audio en cours — peut prendre 1 min...',
    pronunciationScore: 'Confiance en prononciation',
    wordView: 'Vue mot par mot',
    legend: 'Légende',
    legendGood: '≥80% confiant',
    legendWarn: '60–80%',
    legendPoor: '<60% — erreur probable',
    uncertainWords: 'Mots incertains — erreurs de prononciation probables',
    expectedForm: 'Forme attendue',
    likelyError: 'Erreur probable',
    grammaticalRole: 'Rôle grammatical',
    noUncertain: 'Tous les mots prononcés avec haute confiance',
    whisperxBadge: 'WhisperX aligné',
    fallbackBadge: 'Scores Whisper',
    needsArabic: 'Le contrôle de prononciation est disponible pour l\'arabe uniquement.',
    needsSource: 'Générez d\'abord un discours dans le Module A.',
    sourceAudio: 'Audio du discours source',
    yourInterpretation: 'Votre interprétation',
    recordBtn: '🎙 Démarrer l\'enregistrement',
    stopBtn: '⏹ Arrêter l\'enregistrement',
    recordingLive: 'Enregistrement en cours',
    autoTranscribe: 'Transcrire automatiquement après l\'enregistrement',
    orUpload: 'Ou déposez un fichier audio',
    copyText: 'Copier',
    copied: 'Copié !',
    topicPlaceholder: 'Saisissez un sujet ou collez un texte pour générer un discours…',
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
    diffBeginner: 'Débutant',
    diffIntermediate: 'Intermédiaire',
    diffAdvanced: 'Avancé',
  }
};

const NAV_ITEMS = [
  { id: 'module-a', labelKey: 'navA' },
  { id: 'module-b', labelKey: 'navB' },
  { id: 'module-c', labelKey: 'navC' },
  { id: 'module-d', labelKey: 'navD' }
];

const VOICE_OPTIONS = {
  ar: [
    { label: 'Lebanese Arabic — Male (ar-LB)',   accent: 'LB' },
    { label: 'Lebanese Arabic — Female (ar-LB)', accent: 'LB_f' },
    { label: 'Gulf Arabic — Female (ar-SA)',     accent: 'SA' },
    { label: 'Egyptian Arabic — Female (ar-EG)', accent: 'EG' },
    { label: 'Egyptian Arabic — Male (ar-EG)',   accent: 'EG_m' },
  ],
  fr: [
    { label: 'French (Female)',   accent: 'FR' },
    { label: 'French (Male)',     accent: 'FR_m' },
    { label: 'Canadian (Female)', accent: 'CA' },
  ],
  en: [
    { label: 'American (Female)', accent: 'US' },
    { label: 'British (Female)',  accent: 'GB' },
    { label: 'Australian (Female)', accent: 'AU' },
  ]
};

// ── SVG icons ────────────────────────────────────────────────────────────────
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

function TranscriptBubble({ text, vocalizedText, isArabic, onRetranscribe }) {
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
        <button className={`bubble-btn ${copied ? 'bubble-btn-active' : ''}`} onClick={copy} title="Copy">
          <IconCopy />
        </button>
        <button className={`bubble-btn ${feedback === 'up' ? 'bubble-btn-active' : ''}`}
          onClick={() => setFeedback(f => f === 'up' ? null : 'up')} title="Good transcription">
          <IconThumbUp />
        </button>
        <button className={`bubble-btn ${feedback === 'down' ? 'bubble-btn-active' : ''}`}
          onClick={() => setFeedback(f => f === 'down' ? null : 'down')} title="Poor transcription">
          <IconThumbDown />
        </button>
        {onRetranscribe && (
          <button className="bubble-btn" onClick={onRetranscribe} title="Re-transcribe">
            <IconRefresh />
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
  word_count: 200,
  difficulty: 'intermediate',
  mode: 'consecutive',
  structure: 'well-organized',
  number_density: 'low',
  include_hesitations: false,
  pressure_enabled: false,
  speed_pressure: 'normal',
  topic_shifts: 'none',
  context_noise: false,
  cognitive_load: 'medium'
};

function ThreeBackground() {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);

    const globe = new THREE.Mesh(
      new THREE.SphereGeometry(2.2, 64, 64),
      new THREE.MeshStandardMaterial({ color: 0xC8973A, wireframe: true, transparent: true, opacity: 0.32, wireframeLinewidth: 0.015 })
    );
    globe.position.set(1.8, 0, -2);
    globe.scale.setScalar(0.6);
    scene.add(globe);

    const inner = new THREE.Mesh(
      new THREE.SphereGeometry(1.8, 32, 32),
      new THREE.MeshStandardMaterial({ color: 0x2D5A4E, wireframe: true, transparent: true, opacity: 0.32, wireframeLinewidth: 0.015 })
    );
    inner.position.set(1.8, 0, -2);
    inner.scale.setScalar(0.6);
    scene.add(inner);

    const pCount = 120;
    const pGeo = new THREE.BufferGeometry();
    const pPos = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount; i++) {
      pPos[i * 3]     = (Math.random() - 0.5) * 14;
      pPos[i * 3 + 1] = (Math.random() - 0.5) * 10;
      pPos[i * 3 + 2] = (Math.random() - 0.5) * 6 - 3;
    }
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({ color: 0xC8973A, size: 0.025, transparent: true, opacity: 0.5 }));
    scene.add(particles);

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(2.8, 0.012, 8, 100),
      new THREE.MeshBasicMaterial({ color: 0xC8973A, transparent: true, opacity: 0.12 })
    );
    ring.position.set(1.8, 0, -2);
    ring.scale.setScalar(0.6);
    ring.rotation.x = Math.PI / 3;
    scene.add(ring);

    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const pointLight = new THREE.PointLight(0xC8973A, 1.5, 20);
    pointLight.position.set(5, 3, 3);
    scene.add(pointLight);

    let t = 0, rafId;
    const animate = () => {
      rafId = requestAnimationFrame(animate);
      t += 0.005;
      globe.rotation.y = t * 0.3;
      globe.rotation.x = Math.sin(t * 0.2) * 0.15;
      inner.rotation.y = -t * 0.2;
      ring.rotation.z = t * 0.15;
      particles.rotation.y = t * 0.04;
      camera.position.y = Math.sin(t * 0.15) * 0.2;
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);
    return () => { cancelAnimationFrame(rafId); window.removeEventListener('resize', onResize); renderer.dispose(); };
  }, []);
  return <canvas ref={canvasRef} id="three-canvas" />;
}

export default function App() {
  const [uiLang, setUiLang] = useState('en');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [activePanel, setActivePanel] = useState('module-a');
  const [lastGeneratedScript, setLastGeneratedScript] = useState(null);
  const L = UI[uiLang];

  useEffect(() => {
    const isAr = uiLang === 'ar';
    document.documentElement.lang = uiLang;
    document.documentElement.dir = isAr ? 'rtl' : 'ltr';
    document.body.classList.toggle('rtl', isAr);
  }, [uiLang]);

  async function handleLogin(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    const result = await loginUser({
      email: formData.get('email'),
      password: formData.get('password'),
      role: formData.get('role')
    });
    setCurrentUser(result.user);
    setIsAuthenticated(true);
  }

  async function handleSignup(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    const result = await signupUser({
      name: formData.get('name'),
      email: formData.get('email'),
      password: formData.get('password'),
      role: formData.get('role')
    });
    setCurrentUser(result.user);
    setIsAuthenticated(true);
  }

  async function handleLogout() {
    await logoutUser().catch(() => {});
    setIsAuthenticated(false);
    setCurrentUser(null);
    setActivePanel('module-a');
    setLastGeneratedScript(null);
  }

  return (
    <>
      <ThreeBackground />
      <div className="shell">
        <Header
          isAuthenticated={isAuthenticated}
          activePanel={activePanel}
          labels={L}
          onPanelChange={setActivePanel}
          uiLang={uiLang}
          onLanguageChange={setUiLang}
        />
        <main>
          {!isAuthenticated ? (
            <LoginScreen labels={L} onLogin={handleLogin} onSignup={handleSignup} />
          ) : (
            <Workspace
              labels={L}
              activePanel={activePanel}
              onLogout={handleLogout}
              onGenerated={setLastGeneratedScript}
              lastGeneratedScript={lastGeneratedScript}
              currentUser={currentUser}
              isRtl={uiLang === 'ar'}
            />
          )}
        </main>
      </div>
    </>
  );
}

function Header({ isAuthenticated, activePanel, labels, onPanelChange, uiLang, onLanguageChange }) {
  return (
    <header className="fade-up">
      <div>
        <div className="logo-main">ETIB <span>Interpreter</span> Trainer</div>
        <div className="logo-sub">AI Self-Training Platform · USJ Beirut</div>
      </div>
      <div className="arabic-badge">مُدرِّب المُترجِم</div>
      {isAuthenticated && (
        <nav aria-label="Training modules">
          {NAV_ITEMS.map(item => (
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

function LoginScreen({ labels, onLogin, onSignup }) {
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
          <span className="lang-tabs-label">Train in:</span>
          <span className="lang-tab lang-tab-ar">عربي</span>
          <span className="lang-tab lang-tab-fr">Français</span>
          <span className="lang-tab lang-tab-en">English</span>
        </div>
        <h1>Master interpretation with <em>AI-generated</em> speeches</h1>
        <p>An adaptive training platform that generates realistic conference speeches, builds multilingual glossaries, and evaluates your interpretation performance across Arabic, French, and English.</p>
        <div className="metric-row">
          <div className="metric-card">
            <span className="metric-label">Sessions</span>
            <strong className="metric-value">∞</strong>
          </div>
          <div className="metric-card">
            <span className="metric-label">Languages</span>
            <strong className="metric-value">3</strong>
          </div>
          <div className="metric-card">
            <span className="metric-label">Modes</span>
            <strong className="metric-value">3</strong>
            <span className="metric-detail">Consecutive · Simultaneous · Sight</span>
          </div>
        </div>
      </div>

      {/* Right: auth form */}
      <div className="login-panel">
        <h2>{isSignup ? labels.signupTitle : labels.loginTitle}</h2>
        <p className="sub">{isSignup ? labels.signupSubtitle : labels.loginSubtitle}</p>
        <form onSubmit={handleSubmit}>
          {isSignup && (
            <div className="field">
              <label htmlFor="login-name">{labels.name}</label>
              <input id="login-name" name="name" type="text" autoComplete="name" required placeholder="Your full name" />
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
          <div className="field">
            <label htmlFor="login-role">{labels.role}</label>
            <select id="login-role" name="role">
              <option value="student">{labels.student}</option>
              <option value="instructor">{labels.instructor}</option>
              <option value="coordinator">{labels.coordinator}</option>
            </select>
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
        </form>
      </div>
    </section>
  );
}

function Workspace({ labels, activePanel, onLogout, onGenerated, lastGeneratedScript, currentUser, isRtl }) {
  const [sharedAudioUrl, setSharedAudioUrl] = useState(null);
  const [lastTranscript, setLastTranscript] = useState(null);
  const [lastRecordingBlob, setLastRecordingBlob] = useState(null);

  return (
    <section>
      <div className="workspace-header fade-up">
        <span className="workspace-user">
          {currentUser && <><strong>{currentUser.name}</strong> · {currentUser.role}</>}
        </span>
        <button type="button" className="sign-out-btn" onClick={onLogout}>{labels.signOut}</button>
      </div>

      {/* Keep all panels mounted — state persists when switching tabs */}
      <div style={{ display: activePanel === 'module-a' ? 'block' : 'none' }}>
        <ModuleA labels={labels} onGenerated={onGenerated} isRtl={isRtl} />
      </div>
      <div style={{ display: activePanel === 'module-b' ? 'block' : 'none' }}>
        <ModuleB labels={labels} lastGeneratedScript={lastGeneratedScript}
          onAudioGenerated={setSharedAudioUrl} />
      </div>
      <div style={{ display: activePanel === 'module-c' ? 'block' : 'none' }}>
        <ModuleC labels={labels} referenceAudioUrl={sharedAudioUrl}
          sourceScript={lastGeneratedScript?.script || ''}
          onTranscriptComplete={setLastTranscript}
          onRecordingComplete={setLastRecordingBlob} />
      </div>
      <div style={{ display: activePanel === 'module-d' ? 'block' : 'none' }}>
        <ModuleD labels={labels} lastTranscript={lastTranscript}
          lastGeneratedScript={lastGeneratedScript}
          lastRecordingBlob={lastRecordingBlob} />
      </div>

    </section>
  );
}

// ── UN Library Panel ─────────────────────────────────────────────────────────

function UNLibraryPanel({ language, domain, onSelect, onClose }) {
  const [query, setQuery]         = useState('');
  const [results, setResults]     = useState([]);
  const [saved, setSaved]         = useState([]);
  const [tab, setTab]             = useState('search'); // 'search' | 'saved'
  const [status, setStatus]       = useState('idle');   // 'idle'|'searching'|'fetching'
  const [fetchingId, setFetchingId] = useState(null);
  const [error, setError]         = useState('');

  useEffect(() => {
    getSavedSpeeches({ language }).then(d => setSaved(d.saved || [])).catch(() => {});
  }, [language]);

  async function handleSearch(e) {
    e.preventDefault();
    if (!query.trim() && !domain) return;
    setStatus('searching'); setError(''); setResults([]);
    try {
      const data = await searchUNLibrary({ q: query, language, domain, limit: 12 });
      setResults(data.results || []);
      if (!data.results?.length) setError('No results found. Try different keywords.');
    } catch (err) {
      setError(err.message);
    } finally { setStatus('idle'); }
  }

  async function handleFetch(item) {
    setFetchingId(item.un_id); setError('');
    try {
      const data = await fetchUNDocument({
        pdf_url: item.pdf_url, web_url: item.web_url,
        un_id: item.un_id, title: item.title, language,
      });
      setSaved(prev => [{ ...data }, ...prev.filter(s => s.un_id !== data.un_id)]);
      onSelect({ text: data.text, title: data.title, un_id: data.un_id, source: 'UN Digital Library' });
      onClose();
    } catch (err) {
      setError(`Could not load: ${err.message}`);
    } finally { setFetchingId(null); }
  }

  async function handleUseSaved(item) {
    setFetchingId(item.un_id); setError('');
    try {
      const data = await getSavedSpeech(item.un_id);
      onSelect({ text: data.text, title: data.title, un_id: data.un_id, source: 'UN Digital Library' });
      onClose();
    } catch (err) {
      setError(err.message);
    } finally { setFetchingId(null); }
  }

  return (
    <div className="library-overlay">
      <div className="library-panel">
        <div className="library-header">
          <h2 className="library-title">🇺🇳 UN Digital Library</h2>
          <button className="library-close" onClick={onClose}>✕</button>
        </div>
        <p className="library-subtitle">Search real UN speeches to use as grounding for speech generation.</p>

        <div className="library-tabs">
          <button className={`lib-tab ${tab === 'search' ? 'lib-tab-active' : ''}`} onClick={() => setTab('search')}>Search</button>
          <button className={`lib-tab ${tab === 'saved' ? 'lib-tab-active' : ''}`} onClick={() => setTab('saved')}>
            Saved ({saved.length})
          </button>
        </div>

        {tab === 'search' && (
          <>
            <form className="library-search-form" onSubmit={handleSearch}>
              <input
                className="library-search-input"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="e.g. climate change, human rights, economic development…"
              />
              <button type="submit" className="btn-primary" disabled={status === 'searching'}>
                {status === 'searching' ? 'Searching…' : 'Search'}
              </button>
            </form>

            {error && <div className="error-msg" style={{ marginTop: '0.75rem' }}>{error}</div>}

            <div className="library-results">
              {results.map(item => (
                <div key={item.un_id} className="library-result-card">
                  <div className="lib-result-title">{item.title}</div>
                  <div className="lib-result-meta">
                    {item.date && <span>📅 {item.date}</span>}
                    {item.web_url && <a href={item.web_url} target="_blank" rel="noreferrer" className="lib-result-link">View on UN site ↗</a>}
                  </div>
                  {item.description && <p className="lib-result-desc">{item.description}</p>}
                  <button
                    className="btn-primary lib-use-btn"
                    disabled={fetchingId === item.un_id || !item.pdf_url}
                    onClick={() => handleFetch(item)}
                    title={!item.pdf_url ? 'No PDF available for this record' : ''}
                  >
                    {fetchingId === item.un_id ? 'Loading…' : item.pdf_url ? '▷ Use this speech' : 'No PDF'}
                  </button>
                </div>
              ))}
              {status === 'idle' && !results.length && !error && (
                <p className="lib-empty">Search for a topic above to find UN speeches.</p>
              )}
            </div>
          </>
        )}

        {tab === 'saved' && (
          <div className="library-results">
            {saved.length === 0 && <p className="lib-empty">No speeches saved yet. Search and load one first.</p>}
            {saved.map(item => (
              <div key={item.un_id} className="library-result-card">
                <div className="lib-result-title">{item.title}</div>
                <div className="lib-result-meta">
                  {item.word_count && <span>📝 {item.word_count} words</span>}
                  {item.language && <span>🌐 {item.language.toUpperCase()}</span>}
                  {item.saved_at && <span>💾 {new Date(item.saved_at).toLocaleDateString()}</span>}
                </div>
                <button className="btn-primary lib-use-btn"
                  disabled={fetchingId === item.un_id}
                  onClick={() => handleUseSaved(item)}>
                  {fetchingId === item.un_id ? 'Loading…' : '▷ Use this speech'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ModuleA({ labels, onGenerated, isRtl }) {
  const [form, setForm] = useState(initialSpeechForm);
  const [documentFile, setDocumentFile] = useState(null);
  const [retrievalResult, setRetrievalResult] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [showLibrary, setShowLibrary] = useState(false);
  const [librarySource, setLibrarySource] = useState(null); // {text, title, un_id}
  const fileInputRef = useRef(null);
  const isLoading = status === 'loading';

  function updateField(event) {
    const { name, value, type, checked } = event.target;
    setForm(current => ({ ...current, [name]: type === 'checkbox' ? checked : value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setStatus('loading'); setError(''); setResult(null);
    try {
      const data = await generateSpeech({ ...form, word_count: Number(form.word_count) });
      setResult(data); onGenerated(data); setStatus('success');
    } catch (err) { setError(err.message); setStatus('error'); }
  }

  async function handleDocumentGenerate() {
    setStatus('loading'); setError(''); setResult(null); setRetrievalResult(null);
    try {
      if (!documentFile) throw new Error('Choose a TXT, DOCX, or PDF document first.');
      const data = await generateSpeechFromDocument(documentFile, { ...form, word_count: Number(form.word_count) });
      setResult(data); onGenerated(data); setStatus('success');
    } catch (err) { setError(err.message); setStatus('error'); }
  }

  async function handleLibraryGenerate() {
    if (!librarySource?.text) return;
    setStatus('loading'); setError(''); setResult(null);
    try {
      // Convert the extracted UN speech text into a Blob/File and use from-document endpoint
      const blob = new Blob([librarySource.text], { type: 'text/plain' });
      const file = new File([blob], `${librarySource.un_id || 'un-speech'}.txt`, { type: 'text/plain' });
      const data = await generateSpeechFromDocument(file, {
        ...form,
        word_count: Number(form.word_count),
        topic: form.topic || librarySource.title,
      });
      setResult(data); onGenerated(data); setStatus('success');
    } catch (err) { setError(err.message); setStatus('error'); }
  }

  async function handleRetrieveContext() {
    setStatus('loading'); setError(''); setRetrievalResult(null);
    try {
      if (!documentFile) throw new Error('Choose a TXT, DOCX, or PDF document first.');
      const data = await retrieveDocumentContext(documentFile, {
        query: form.topic, language: form.language, domain: form.domain,
        scenario: 'UN General Assembly', difficulty: form.difficulty,
        mode: form.mode, number_density: form.number_density, max_chunks: 4
      });
      setRetrievalResult(data); setStatus('success');
    } catch (err) { setError(err.message); setStatus('error'); }
  }

  const LANG_LABEL = { ar: 'AR', fr: 'FR', en: 'EN' };

  return (
    <div className="card module-a-card">
      <form onSubmit={handleSubmit}>

        {/* ── Main topic bar ── */}
        <div className="topic-bar">
          <input
            className="topic-input"
            name="topic"
            type="text"
            value={form.topic}
            minLength="3"
            maxLength="180"
            required
            placeholder={labels.topicPlaceholder || 'Enter a topic or paste text to generate a speech…'}
            onChange={updateField}
            disabled={isLoading}
          />
          <button
            type="button"
            className="topic-attach-btn"
            title="Attach a document (PDF, DOCX, TXT)"
            onClick={() => fileInputRef.current?.click()}
          >
            {documentFile ? '📄' : '+'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.docx,.pdf"
            style={{ display: 'none' }}
            onChange={event => { setDocumentFile(event.target.files?.[0] || null); setRetrievalResult(null); }}
          />
        </div>

        {/* ── Attached file chip ── */}
        {documentFile && (
          <div className="file-chip">
            <span>📄 {documentFile.name}</span>
            <button type="button" className="file-chip-remove" onClick={() => { setDocumentFile(null); setRetrievalResult(null); }}>×</button>
          </div>
        )}

        {/* ── UN Library source chip ── */}
        {librarySource && (
          <div className="file-chip file-chip-un">
            <span>🇺🇳 {librarySource.title.slice(0, 60)}{librarySource.title.length > 60 ? '…' : ''}</span>
            <button type="button" className="file-chip-remove" onClick={() => setLibrarySource(null)}>×</button>
          </div>
        )}

        {/* ── Language + quick settings row ── */}
        <div className="quick-settings-row">
          <div className="lang-pair">
            <select name="language" value={form.language} onChange={updateField} className="lang-select" title="Speech language">
              <option value="ar">AR</option>
              <option value="fr">FR</option>
              <option value="en">EN</option>
            </select>
            <span className="lang-arrow">{isRtl ? '←' : '→'}</span>
            <select name="target_language" value={form.target_language} onChange={updateField} className="lang-select" title="Interpretation target">
              <option value="ar">AR</option>
              <option value="fr">FR</option>
              <option value="en">EN</option>
            </select>
          </div>

          <select name="domain" value={form.domain} onChange={updateField} className="quick-select">
            <option value="politics">{labels.domPolitics || 'Politics'}</option>
            <option value="diplomacy">{labels.domDiplomacy || 'Diplomacy'}</option>
            <option value="economics">{labels.domEconomics || 'Economics'}</option>
            <option value="health">{labels.domHealth || 'Health'}</option>
            <option value="education">{labels.domEducation || 'Education'}</option>
          </select>

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

        {/* ── Advanced options (collapsible) ── */}
        {showAdvanced && (
          <div className="advanced-panel">
            <div className="form-grid">
              <div className="field">
                <label htmlFor="f-words">{labels.wordCount}</label>
                <input id="f-words" name="word_count" type="number" value={form.word_count} min="50" max="800" onChange={updateField} />
              </div>
              <SelectField label={labels.mode} id="f-mode" name="mode" value={form.mode} onChange={updateField}>
                <option value="consecutive">{labels.optConsecutive || 'Consecutive'}</option>
                <option value="simultaneous">{labels.optSimultaneous || 'Simultaneous'}</option>
                <option value="sight_translation">{labels.optSight || 'Sight translation'}</option>
              </SelectField>
              <SelectField label={labels.structure} id="f-structure" name="structure" value={form.structure} onChange={updateField}>
                <option value="well-organized">{labels.optWellOrganized || 'Well organized'}</option>
                <option value="deliberately disorganized">{labels.optDisorganized || 'Disorganized'}</option>
              </SelectField>
              <SelectField label={labels.numbers} id="f-numbers" name="number_density" value={form.number_density} onChange={updateField}>
                <option value="low">{labels.optLowNumbers || 'Low numbers'}</option>
                <option value="high">{labels.optHighNumbers || 'High numbers'}</option>
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
            </div>
          </div>
        )}

        {/* ── Action buttons ── */}
        <div className="action-row">
          <button type="button" className="btn-un-library" onClick={() => setShowLibrary(true)} disabled={isLoading}>
            🇺🇳 UN Library
          </button>
          {librarySource ? (
            <button type="button" className="btn-primary" onClick={handleLibraryGenerate} disabled={isLoading}>
              {isLoading ? labels.generating : '▷ Generate from UN speech'}
            </button>
          ) : documentFile ? (
            <>
              <button type="button" className="btn-secondary" onClick={handleRetrieveContext} disabled={isLoading}>
                Preview context
              </button>
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
      {retrievalResult && <RetrievalResult data={retrievalResult} labels={labels} />}
      {result && <SpeechResult data={result} labels={labels} />}

      {showLibrary && (
        <UNLibraryPanel
          language={form.language}
          domain={form.domain}
          onSelect={src => { setLibrarySource(src); setDocumentFile(null); }}
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

function RetrievalResult({ data, labels }) {
  return (
    <section className="retrieval-result">
      <h3>{labels.retrievedContext || 'Retrieved context'}</h3>
      <div className="result-meta">
        <span>{data.selected_chunk_count || 0} chunks</span>
        {(data.documents_processed || []).map((doc, index) => (
          <span key={`${doc.filename}-${index}`}>{doc.filename}</span>
        ))}
      </div>
      <div className="chunk-list">
        {(data.selected_chunks || []).map((chunk, index) => (
          <article className="chunk-item" key={`${chunk.source_filename}-${chunk.chunk_index}-${index}`}>
            <p className="chunk-source">
              {chunk.source_filename} · chunk {Number(chunk.chunk_index || 0) + 1}
            </p>
            <p>{chunk.text}</p>
          </article>
        ))}
      </div>
      {Array.isArray(data.document_errors) && data.document_errors.length > 0 && (
        <div className="error-msg">
          {data.document_errors.map((item, index) => (
            <p key={`${item.filename}-${index}`}>{item.filename}: {item.error}</p>
          ))}
        </div>
      )}
    </section>
  );
}

function SpeechResult({ data, labels }) {
  const duration = useMemo(() => {
    const seconds = Number(data.estimated_duration_seconds || 0);
    if (!seconds) return null;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  }, [data.estimated_duration_seconds]);

  const isArabic = data.language === 'ar';

  return (
    <div className="speech-result">
      <div className="result-meta">
        <span>{data.word_count} {labels.wordsUnit}</span>
        {duration && <span>~{duration}</span>}
        {data.topic && <span>{data.topic}</span>}
        <span>{data.domain}</span>
        <span>{String(data.language || '').toUpperCase()} → {String(data.target_language || '').toUpperCase()}</span>
      </div>
      {data.mode === 'un_library_grounded' && data.source_speech && (
        <p className="grounded-source-note">
          {labels.groundedSourceLabel}{' '}
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
      {data.mode === 'generated' && (
        <p className="grounded-source-note grounded-source-note--missing">
          {labels.ungroundedNote}
        </p>
      )}
      <div className={`speech-text ${isArabic ? 'arabic' : ''}`}>
        {data.script}
      </div>
      <p className="speech-next-hint">→ Go to <strong>Audio &amp; materials</strong> to listen, review questions, and download the glossary.</p>
    </div>
  );
}

// ── Module B — TTS + Pedagogical Materials ──────────────────────────────────

function ModuleB({ labels, lastGeneratedScript, onAudioGenerated }) {
  const language = lastGeneratedScript?.language || 'ar';
  const isArabic = language === 'ar';
  const voiceOptions = VOICE_OPTIONS[language] || VOICE_OPTIONS.en;
  const speechId = lastGeneratedScript?.script?.slice(0, 50) || '';

  const [selectedAccent, setSelectedAccent] = useState(voiceOptions[0]?.accent || '');
  const [speechRate, setSpeechRate] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioStatus, setAudioStatus] = useState('idle');
  const [audioError, setAudioError] = useState('');
  const [keyTerms, setKeyTerms] = useState(null);
  const [keyTermsLoading, setKeyTermsLoading] = useState(false);
  const [keyTermsError, setKeyTermsError] = useState('');

  useEffect(() => {
    const opts = VOICE_OPTIONS[language] || VOICE_OPTIONS.en;
    setSelectedAccent(opts[0]?.accent || '');
    setAudioUrl(null);
    setAudioStatus('idle');
    setAudioError('');
    setKeyTerms(null);
    setKeyTermsLoading(false);
    setKeyTermsError('');
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
      const url = `http://127.0.0.1:5000${data.audio_url}`;
      setAudioUrl(url); onAudioGenerated?.(url); setAudioStatus('success');
    } catch (err) { setAudioError(err.message); setAudioStatus('error'); }
  }

  async function handleGenerateKeyTerms() {
    setKeyTermsLoading(true); setKeyTermsError('');
    try {
      const data = await generateMaterials({ script: lastGeneratedScript.script, language, domain: lastGeneratedScript.domain });
      setKeyTerms(data.key_terms || []);
    } catch (err) { setKeyTermsError(err.message); setKeyTerms([]); }
    finally { setKeyTermsLoading(false); }
  }

  async function handleDownloadGlossary() {
    try {
      const mapped = glossary.map(item => ({
        ar: item.arabic || item.ar || '',
        fr: item.french || item.fr || '',
        en: item.english || item.en || '',
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

      {/* ── Audio ── */}
      <div className="card">
        <h2 className="b-section-title">🔊 {labels.moduleBTitle}</h2>
        <div className="form-grid" style={{ marginBottom: '1rem' }}>
          <SelectField label={labels.voiceAccent} id="b-voice" name="voice"
            value={selectedAccent} onChange={e => setSelectedAccent(e.target.value)}>
            {voiceOptions.map(v => <option key={v.accent} value={v.accent}>{v.label}</option>)}
          </SelectField>
          <div className="field">
            <label htmlFor="b-rate">{labels.speechRate}: {speechRate > 0 ? `+${speechRate}` : speechRate}%</label>
            <input id="b-rate" type="range" min="-30" max="20" step="5"
              value={speechRate} onChange={e => setSpeechRate(Number(e.target.value))} className="rate-slider" />
          </div>
        </div>
        <button className="btn-primary" onClick={handleGenerateAudio} disabled={audioStatus === 'loading'}>
          {audioStatus === 'loading' ? labels.generatingAudio : `▷ ${labels.generateAudio}`}
        </button>
        {audioError && <div className="error-msg" style={{ marginTop: '0.75rem' }}>{labels.errorPrefix}: {audioError}</div>}
        {audioUrl && <div style={{ marginTop: '1rem' }}><audio key={audioUrl} controls src={audioUrl} style={{ width: '100%' }} /></div>}
      </div>

      {/* ── Summary ── */}
      {summary && (
        <div className="card">
          <h2 className="b-section-title">📋 {labels.summaryTitle}</h2>
          <pre className="mind-map">{summary}</pre>
        </div>
      )}

      {/* ── MCQ ── */}
      {mcqs.length > 0 && (
        <div className="card">
          <h2 className="b-section-title">❓ {labels.mcqTitle}</h2>
          <div className="table-responsive">
            <table className="mcq-table">
              <thead>
                <tr><th style={{width:'2.5rem'}}>#</th><th>Question</th><th>Options</th><th style={{width:'5rem'}}>Answer</th></tr>
              </thead>
              <tbody>
                {mcqs.map((item, i) => (
                  <tr key={i}>
                    <td className="mcq-num">{i + 1}</td>
                    <td className={`mcq-q ${isArabic ? 'arabic' : ''}`}>{item.question}</td>
                    <td>
                      <ul className="mcq-opts-inline">
                        {(item.options || []).map((opt, oi) => (
                          <li key={oi} className={
                            item.answer && (opt === item.answer || opt.startsWith(item.answer + '.') || opt.startsWith(item.answer + ' '))
                              ? 'mcq-correct-opt' : ''
                          }>{opt}</li>
                        ))}
                      </ul>
                    </td>
                    <td className="mcq-ans">{item.answer}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Glossary ── */}
      {glossary.length > 0 && (
        <div className="card">
          <div className="b-section-header">
            <h2 className="b-section-title">📖 {labels.glossaryTitle}</h2>
            <button className="btn-secondary btn-sm" onClick={handleDownloadGlossary}>{labels.downloadGlossary}</button>
          </div>
          <div className="table-responsive">
            <table className="glossary-table">
              <thead>
                <tr><th>Term</th><th>العربية</th><th>Français</th><th>English</th><th>Definition</th></tr>
              </thead>
              <tbody>
                {glossary.map((item, i) => (
                  <tr key={i}>
                    <td><strong>{item.term}</strong></td>
                    <td className="arabic" dir="rtl">{item.arabic || item.ar || ''}</td>
                    <td>{item.french || item.fr || ''}</td>
                    <td>{item.english || item.en || ''}</td>
                    <td className="gloss-def">{item.definition || ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Key Terms (on demand) ── */}
      <div className="card">
        <h2 className="b-section-title">🔑 {labels.keyTerms}</h2>
        <button className="btn-secondary" onClick={handleGenerateKeyTerms} disabled={keyTermsLoading}>
          {keyTermsLoading ? labels.generatingMaterials : labels.generateMaterials}
        </button>
        {keyTermsError && <div className="error-msg" style={{ marginTop: '0.5rem' }}>{keyTermsError}</div>}
        {keyTerms && keyTerms.length > 0 && (
          <div className="key-terms-list">
            {keyTerms.map((term, i) => (
              <span key={i} className={`key-term-badge ${isArabic ? 'arabic' : ''}`}>{term}</span>
            ))}
          </div>
        )}
      </div>

    </div>
  );
}

// ── Module C — ASR Transcription + Browser Recording ────────────────────────

function ModuleC({ labels, referenceAudioUrl, sourceScript, onTranscriptComplete, onRecordingComplete }) {
  const [language, setLanguage] = useState('ar');
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [autoTranscribe, setAutoTranscribe] = useState(true);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

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
      setResult(data);
      onTranscriptComplete?.({ ...data, language, sourceScript: sourceScript || '' });
      setStatus('success');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus' : 'audio/webm';
      const mr = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      setRecordingTime(0);
      setResult(null);
      setError('');

      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        stream.getTracks().forEach(t => t.stop());
        clearInterval(timerRef.current);
        setRecordedBlob(blob);
        onRecordingComplete?.(blob);
        if (recordedUrl) URL.revokeObjectURL(recordedUrl);
        setRecordedUrl(URL.createObjectURL(blob));
        if (autoTranscribe) runTranscription(blob);
      };

      mr.start(250);
      setIsRecording(true);
      timerRef.current = setInterval(() => setRecordingTime(t => t + 1), 1000);
    } catch (err) {
      setError('Microphone access denied — please allow microphone permission and try again.');
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
    clearInterval(timerRef.current);
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

        {/* Reference audio from Module B */}
        {referenceAudioUrl && (
          <div className="reference-audio-box">
            <p className="ref-audio-label">🔊 {labels.sourceAudio}</p>
            <audio controls src={referenceAudioUrl} style={{ width: '100%' }} />
          </div>
        )}

        {/* Recording section */}
        <div className="record-section">
          <p className="record-section-label">🎙 {labels.yourInterpretation}</p>

          <div className="record-controls">
            {!isRecording ? (
              <button className="btn-record" onClick={startRecording}>
                {labels.recordBtn}
              </button>
            ) : (
              <button className="btn-record recording-active" onClick={stopRecording}>
                <span className="rec-dot" /> {labels.stopBtn} — {formatTime(recordingTime)}
              </button>
            )}

            <label className="auto-transcribe-toggle">
              <input type="checkbox" checked={autoTranscribe}
                onChange={e => setAutoTranscribe(e.target.checked)} />
              {labels.autoTranscribe}
            </label>
          </div>

          {/* Recorded audio playback */}
          {recordedUrl && !isRecording && (
            <div style={{ marginTop: '0.75rem' }}>
              <audio controls src={recordedUrl} style={{ width: '100%' }} />
              {!autoTranscribe && (
                <button className="btn-primary" style={{ marginTop: '0.75rem' }}
                  onClick={() => runTranscription(recordedBlob)}
                  disabled={status === 'loading'}>
                  {status === 'loading' ? labels.transcribing : labels.transcribeBtn}
                </button>
              )}
            </div>
          )}
        </div>

        {/* Upload fallback */}
        <details className="upload-fallback">
          <summary>{labels.orUpload}</summary>
          <div style={{ marginTop: '0.75rem' }}>
            <input type="file" accept=".mp3,.wav,.m4a,.ogg,.webm" className="file-input"
              onChange={e => {
                const f = e.target.files[0];
                if (f) {
                  setRecordedBlob(f);
                  onRecordingComplete?.(f);
                  runTranscription(f);
                }
              }} />
          </div>
        </details>
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
                {result.method && <span className="method-badge">{result.method}</span>}
              </div>

              {lowConfWords.length > 0 && (
                <div className="warning-msg" style={{ marginBottom: '0.75rem' }}>
                  ⚠️ {labels.lowConfidence}: {lowConfWords.slice(0, 10).map(w => w.word.trim()).join(', ')}
                </div>
              )}

              <TranscriptBubble
                text={result.full_text}
                vocalizedText={result.vocalized_text}
                isArabic={isArabic}
                onRetranscribe={recordedBlob ? () => runTranscription(recordedBlob) : null}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Module D — Full Evaluation Report ────────────────────────────────────────

function ScoreBar({ score }) {
  const pct = Math.round((score / 10) * 100);
  const color = score >= 7 ? '#2D5A4E' : score >= 5 ? '#C8973A' : '#8B3A2A';
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
        <div style={{ fontSize: '0.74rem', color: 'var(--warm-gray)', marginTop: '0.3rem' }}>{pct}% — {score >= 7 ? 'Good' : score >= 5 ? 'Acceptable' : 'Needs work'}</div>
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
    ? (result.coverage_score >= 8 ? '#1a6b3c' : result.coverage_score >= 6 ? '#b0772f' : '#c0392b')
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
              <p className="report-label">Coverage score</p>
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
              <h4 style={{ color: '#b0772f', fontSize: '0.85rem', fontWeight: 700,
                textTransform: 'uppercase', marginBottom: '0.6rem' }}>
                ⚠️ {labels.languageErrors} — إعراب / تشكيل ({result.tashkeel_errors.length})
              </h4>
              {result.tashkeel_errors.map((err, i) => (
                <div key={i} className="uncertain-word-card">
                  <div className="uncertain-word-header">
                    <span className="uncertain-word arabic">{err.word}</span>
                    {err.expected_form && (
                      <span className="detail-label" style={{ marginLeft: '0.5rem' }}>
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
                ✅ Correct tashkeel ({result.tashkeel_correct.length})
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

function ModuleD({ labels, lastTranscript, lastGeneratedScript, lastRecordingBlob }) {
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');

  const language = lastTranscript?.language || lastTranscript?.language_detected || 'ar';

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
          lastGeneratedScript?.language || language
        );
      } else {
        // Fallback: use stored Groq transcript (less accurate for hesitations)
        data = await generateFeedback({
          source_script:   lastGeneratedScript?.script || '',
          transcript_text: lastTranscript.full_text,
          transcript:      lastTranscript,
          language
        });
      }
      // Attach pronunciation report for the pronunciation panel
      if (data.pronunciation) {
        data._pronunciationForPanel = {
          overall_score: data.pronunciation.overall_score,
          uncertain_count: data.pronunciation.uncertain_count,
          flagged_words: data.pronunciation.flagged_words || [],
          diff_errors: data.pronunciation.diff_errors || [],
          summary: data.pronunciation.summary,
        };
      }
      setReport(data);
      setStatus('success');
    } catch (err) {
      setError(err.message);
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
  const isAr = (lastTranscript.language || lastTranscript.language_detected) === 'ar';

  return (
    <div>
      <div className="card">
        <h2>{labels.moduleDTitle}</h2>
        <div className="info-tip" style={{ marginBottom: '1rem' }}>
          📄 Transcript ready ({lastTranscript.duration_seconds}s · {(lastTranscript.language_detected || '').toUpperCase()})
          {lastGeneratedScript && <> · Source: {lastGeneratedScript.domain}</>}
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

          {/* Score + Summary */}
          <div className="card report-header">
            <div className="report-score-row">
              <div>
                <p className="report-label">{labels.overallScore}</p>
                <ScoreBar score={report.overall_score || 0} />
              </div>
              {(report.strengths || []).length > 0 && (
                <div className="report-strengths">
                  <p className="report-label">✅ {labels.strengths}</p>
                  <ul>{(report.strengths || []).map((s, i) => <li key={i}>{s}</li>)}</ul>
                </div>
              )}
            </div>
            {report.summary && (
              <div className="report-summary">
                <p className="report-label">📋 {labels.evalSummary}</p>
                <p className={isAr ? 'arabic' : ''}>{report.summary}</p>
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
                { label: labels.numberErrors,  icon: '🔢', items: algo.number_errors || [] },
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
                {algo.long_silences.map((s, i) => (
                  <div key={i} className="algo-detail-row">
                    <span className="algo-detail-badge algo-badge-warn">{s.duration_seconds}s</span>
                    <span className="algo-detail-text">at {s.at_seconds}s{s.after_text ? ` after: "${s.after_text}"` : ''}</span>
                  </div>
                ))}
              </div>
            )}
            {(algo.repetitions || []).length > 0 && (
              <div className="algo-detail-block">
                <p className="algo-detail-title">🔁 {labels.repetitions}</p>
                {algo.repetitions.map((r, i) => (
                  <div key={i} className="algo-detail-row">
                    <span className="algo-detail-badge algo-badge-gold">"{r.word}"</span>
                    <span className="algo-detail-text">at {r.at_seconds}s · repeated at {r.second_occurrence}s</span>
                  </div>
                ))}
              </div>
            )}
            {(algo.hesitation_words || []).length > 0 && (
              <div className="algo-detail-block">
                <p className="algo-detail-title">🗣️ {labels.hesitations}</p>
                <div className="algo-chips">
                  {algo.hesitation_words.map((h, i) => (
                    <span key={i} className="algo-chip">
                      "{h.word}" <small>{h.at_seconds !== undefined ? `${h.at_seconds}s` : ''}</small>
                    </span>
                  ))}
                </div>
              </div>
            )}
            {(algo.number_errors || []).length > 0 && (
              <div className="algo-detail-block">
                <p className="algo-detail-title">🔢 {labels.numberErrors}</p>
                <div className="algo-chips">
                  {algo.number_errors.map((n, i) => (
                    <span key={i} className="algo-chip algo-chip-err">{n}</span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Coverage score */}
          {report.coverage_score !== undefined && (
            <div className="card">
              <h3 className="report-section-title">📊 {labels.coverageTitle}</h3>
              <ScoreBar score={report.coverage_score} />
              <p style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--warm-gray)' }}>
                {labels.coverageHint}
              </p>
            </div>
          )}

          {/* Translation errors */}
          {(report.translation_errors || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">🔄 {labels.translationErrors}</h3>
              {(report.translation_errors || []).map((item, i) => (
                <div key={i} className="eval-item" style={{ borderLeft: '3px solid var(--sienna)', paddingLeft: '0.75rem', marginBottom: '0.75rem' }}>
                  <div style={{ fontSize: '0.78rem', color: 'var(--warm-gray)', marginBottom: '0.2rem' }}>{labels.sourceSaid}:</div>
                  <div className="eval-text" style={{ marginBottom: '0.25rem' }}>"{item.source_text}"</div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--warm-gray)', marginBottom: '0.2rem' }}>{labels.studentSaid}: <strong style={{ color: 'var(--sienna)' }}>{item.student_said}</strong></div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--warm-gray)', marginBottom: '0.2rem' }}>{labels.correctTranslation}: <strong style={{ color: 'var(--sage)' }}>{item.correct_translation}</strong></div>
                  {item.explanation && <div className="eval-explanation">{item.explanation}</div>}
                </div>
              ))}
            </div>
          )}

          {/* Missing content */}
          {(report.missing_content || []).length > 0 && (
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

          {/* Pronunciation Report (EN/FR: full analysis; AR: confidence only) */}
          {report.pronunciation && (report.pronunciation.uncertain_count > 0 || (report.pronunciation.diff_errors || []).length > 0) && (
            <div className="card">
              <h3 className="report-section-title">🔊 {labels.pronunciationTitle}</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.8rem', flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--warm-gray)', marginBottom: '0.3rem' }}>{labels.pronunciationScore}</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, fontFamily: 'Playfair Display, serif', color: report.pronunciation.overall_score >= 0.8 ? 'var(--sage)' : report.pronunciation.overall_score >= 0.65 ? 'var(--gold)' : 'var(--sienna)' }}>
                    {Math.round(report.pronunciation.overall_score * 100)}%
                  </div>
                </div>
                <p style={{ fontSize: '0.83rem', color: 'var(--warm-gray)', maxWidth: 340 }}>{report.pronunciation.summary}</p>
              </div>

              {/* Flagged words with notes */}
              {(report.pronunciation.flagged_words || []).length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <p style={{ fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--warm-gray)', marginBottom: '0.5rem' }}>{labels.uncertainWords}</p>
                  <div className="word-grid">
                    {(report.pronunciation.flagged_words || []).map((w, i) => (
                      <span key={i} className="word-chip word-poor" title={w.note || ''} style={{ cursor: w.note ? 'help' : 'default' }}>
                        {w.word} <small style={{ opacity: 0.7 }}>{Math.round(w.score * 100)}%</small>
                      </span>
                    ))}
                  </div>
                  {(report.pronunciation.flagged_words || []).filter(w => w.note).slice(0, 3).map((w, i) => (
                    <div key={i} className="eval-item" style={{ borderLeft: '3px solid var(--gold)', paddingLeft: '0.6rem', marginTop: '0.4rem' }}>
                      <strong style={{ fontSize: '0.84rem' }}>"{w.word}"</strong>
                      <div className="eval-explanation">{w.note}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Reference diffing errors (EN/FR) */}
              {(report.pronunciation.diff_errors || []).length > 0 && (
                <div>
                  <p style={{ fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--warm-gray)', marginBottom: '0.5rem' }}>{labels.pronunciationMismatches}</p>
                  {(report.pronunciation.diff_errors || []).slice(0, 5).map((e, i) => (
                    <div key={i} className="eval-item" style={{ borderLeft: '3px solid var(--sienna)', paddingLeft: '0.6rem', marginBottom: '0.4rem' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--warm-gray)' }}>{e.type}: </span>
                      {e.expected && <span style={{ fontSize: '0.84rem' }}>expected <strong style={{ color: 'var(--sage)' }}>"{e.expected}"</strong></span>}
                      {e.said && <span style={{ fontSize: '0.84rem' }}> · said <strong style={{ color: 'var(--sienna)' }}>"{e.said}"</strong></span>}
                      {e.note && <div className="eval-explanation">{e.note}</div>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* LLM pronunciation flags (from LLM analysis) */}
          {(report.pronunciation_flags || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">🗣 AI pronunciation analysis</h3>
              {(report.pronunciation_flags || []).map((item, i) => (
                <div key={i} className="eval-item" style={{ borderLeft: '3px solid var(--gold)', paddingLeft: '0.75rem', marginBottom: '0.5rem' }}>
                  <strong>"{item.word}"</strong>
                  {item.confidence !== undefined && <span style={{ fontSize: '0.78rem', color: 'var(--warm-gray)', marginLeft: '0.5rem' }}>{Math.round(item.confidence * 100)}% confidence</span>}
                  {item.likely_issue && <div className="eval-explanation">{item.likely_issue}</div>}
                </div>
              ))}
            </div>
          )}

          {/* LLM analysis sections */}
          <div className="card">
            <h3 className="report-section-title">🤖 {labels.llmTitle}</h3>
            <EvalSection title={labels.languageErrors} icon="⚠️"
              items={report.language_errors || []} emptyLabel={labels.noIssues}
              renderItem={item => (
                <div>
                  <span className={`eval-text ${isAr ? 'arabic' : ''}`}>"{item.text}"</span>
                  <span className="eval-explanation"> — {item.explanation}</span>
                  {item.correction && <div className="eval-correction">✓ {labels.correction}: <strong>{item.correction}</strong></div>}
                </div>
              )} />
            <EvalSection title={labels.autoCorrections} icon="✏️"
              items={report.auto_corrections || []} emptyLabel={labels.noIssues}
              renderItem={item => <span className={isAr ? 'arabic' : ''}>{item.text}</span>} />
            <EvalSection title={labels.falseStarts} icon="🚦"
              items={report.false_starts || []} emptyLabel={labels.noIssues}
              renderItem={item => <span className={isAr ? 'arabic' : ''}>{item.text}</span>} />
            <EvalSection title={labels.lapsusLinguae} icon="👅"
              items={report.lapsus_linguae || []} emptyLabel={labels.noIssues}
              renderItem={item => (
                <div>
                  <span className={`eval-text ${isAr ? 'arabic' : ''}`}>"{item.text}"</span>
                  {item.likely_intended && <span className="eval-explanation"> → {item.likely_intended}</span>}
                </div>
              )} />
            <EvalSection title={labels.terminologyProblems} icon="📚"
              items={report.terminology_problems || []} emptyLabel={labels.noIssues}
              renderItem={item => (
                <div className="term-row">
                  <span className="term-cell"><small>{labels.source}:</small> {item.source_term}</span>
                  <span className="term-cell"><small>{labels.used}:</small> <strong className="term-wrong">{item.student_used}</strong></span>
                  <span className="term-cell"><small>{labels.correct}:</small> <strong className="term-right">{item.correct_equivalent}</strong></span>
                </div>
              )} />
            <EvalSection title={labels.informationLoss} icon="📉"
              items={report.information_loss || []} emptyLabel={labels.noIssues}
              renderItem={item => (
                <div>
                  <span className="eval-text">{item.lost_content}</span>
                  {item.importance && <span className={`importance-badge importance-${item.importance}`}>{item.importance}</span>}
                </div>
              )} />
          </div>

          {/* Recommendations */}
          {(report.recommendations || []).length > 0 && (
            <div className="card">
              <h3 className="report-section-title">💡 {labels.recommendations}</h3>
              <ul className="recommendations-list">
                {report.recommendations.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Pronunciation Assessment — Arabic only */}
      {lastTranscript && (lastTranscript.language || lastTranscript.language_detected) === 'ar' && (
        <div className="card" style={{ marginTop: '1rem' }}>
          <h3 className="report-section-title">🔤 {labels.pronunciationTitle}</h3>
          <PronunciationPanel
            labels={labels}
            lastTranscript={lastTranscript}
            lastGeneratedScript={lastGeneratedScript}
          />
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
