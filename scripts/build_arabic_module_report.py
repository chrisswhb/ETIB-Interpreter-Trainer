from __future__ import annotations

import html
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ETIB_Arabic_Module_Detailed_Report.docx"
HTML_OUT = ROOT / "ETIB_Arabic_Module_Detailed_Report.html"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def run(text: str, bold: bool = False, size: int | None = None, rtl: bool = False, font: str | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    if rtl:
        props.append("<w:rtl/>")
    if font:
        props.append(f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    parts = str(text).split("\n")
    body = []
    for index, part in enumerate(parts):
        if index:
            body.append("<w:br/>")
        body.append(f"<w:t xml:space=\"preserve\">{esc(part)}</w:t>")
    return f"<w:r>{rpr}{''.join(body)}</w:r>"


def paragraph(text: str = "", style: str | None = None, bold: bool = False, rtl: bool = False, mono: bool = False) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if rtl:
        ppr.append("<w:bidi/>")
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    return f"<w:p>{ppr_xml}{run(text, bold=bold, rtl=rtl, font='Courier New' if mono else None)}</w:p>"


def bullet(text: str, rtl: bool = False) -> str:
    ppr = '<w:pPr><w:pStyle w:val="ListParagraph"/><w:ind w:left="720" w:hanging="360"/></w:pPr>'
    return f"<w:p>{ppr}{run('• ', rtl=rtl)}{run(text, rtl=rtl)}</w:p>"


def heading(text: str, level: int = 1) -> str:
    style = "Heading1" if level == 1 else "Heading2"
    return paragraph(text, style=style, bold=True)


def cell(content: str, shade: str | None = None) -> str:
    shd = f'<w:shd w:fill="{shade}"/>' if shade else ""
    return (
        "<w:tc><w:tcPr>"
        '<w:tcW w:w="2400" w:type="dxa"/>'
        f"{shd}</w:tcPr>"
        f"{content}"
        "</w:tc>"
    )


def table(rows: list[list[str]], header: bool = True) -> str:
    xml = [
        "<w:tbl>",
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/>'
        "</w:tblBorders></w:tblPr>",
    ]
    for i, row in enumerate(rows):
        xml.append("<w:tr>")
        for value in row:
            xml.append(cell(paragraph(value, bold=(header and i == 0)), shade="DDEBF7" if header and i == 0 else None))
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def document_xml() -> str:
    body: list[str] = []

    body.append(paragraph("ETIB Arabic Module: Pronunciation, Tashkeel and Tanween Error Detection", "Title", bold=True))
    body.append(paragraph("Detailed technical report and handoff document"))
    body.append(paragraph("Date: 30 June 2026"))
    body.append(paragraph("Project path: C:\\Users\\Ali\\Downloads\\etib_project\\etib", mono=True))
    body.append(paragraph("Scope: Arabic evaluation component only; not the full interpreter training platform."))

    body.append(heading("1. Executive Summary"))
    body.append(paragraph(
        "This module evaluates Arabic spoken responses by comparing the learner's recording against a reference Arabic text. "
        "The focus is final-word pronunciation: tashkeel endings, i'rab vowels, tanween, and related pronunciation signals. "
        "The system was upgraded from a local ASR-only approach to a reference-guided multi-signal architecture using Groq Whisper hypothesis scoring, "
        "a Groq-based Arabic diacritizer, a three-model wav2vec2 ensemble, word-aware LLR acoustic scoring, and confidence-based decision fusion."
    ))
    body.append(paragraph(
        "The practical target is >70% accuracy on controlled or clean audio when the reference is fully diacritized or automatically diacritized first. "
        "A robust 80%+ claim on arbitrary Arabic text is possible only if the reference text contains correct final endings and audio quality is reasonable; "
        "without those conditions, Arabic ASR models often omit short vowels and tanween, making the problem acoustically difficult."
    ))

    body.append(heading("2. Tools and Technologies"))
    body.append(table([
        ["Layer", "Tool / Library", "Purpose"],
        ["Frontend", "HTML, CSS, JavaScript MediaRecorder", "Record audio in the browser, choose exercises, paste custom Arabic text, display findings."],
        ["Backend", "FastAPI + Uvicorn", "Expose REST endpoints for exercise loading, audio analysis, Groq status, and diacritization."],
        ["Audio processing", "ffmpeg through imageio-ffmpeg fallback", "Convert browser WebM/Opus audio to 16 kHz mono WAV."],
        ["Local ASR", "Hugging Face Transformers + PyTorch", "Run Arabic wav2vec2 CTC models locally."],
        ["VAD / delivery", "Silero VAD + numpy/scipy style audio processing", "Detect pauses, duration, speech rate, and basic delivery metrics."],
        ["Cloud ASR scoring", "Groq Whisper API", "Score competing pronunciation hypotheses using avg_logprob."],
        ["Diacritization", "Groq LLaMA model", "Convert unvocalized Arabic reference text into fully diacritized text before evaluation."],
        ["Model storage", "Desktop cache folder", "Keep models installed once so the app starts without re-downloading them."],
    ]))

    body.append(heading("3. Model Inventory"))
    body.append(table([
        ["Model / Component", "Local folder or API name", "Role in architecture"],
        ["jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "Desktop\\etib_models\\wav2vec2_jonatasgrosman_arabic", "Arabic CTC transcript signal; one vote in the ensemble."],
        ["elgeish/wav2vec2-large-xlsr-53-arabic", "Desktop\\etib_models\\wav2vec2_elgeish_arabic", "Second Arabic CTC transcript signal and logits for LLR."],
        ["KMFODA Arabic wav2vec2", "Desktop\\etib_models\\wav2vec2_kmfoda_arabic", "Third Arabic CTC transcript signal and logits for LLR."],
        ["Silero VAD", "Desktop\\etib_models\\silero-vad", "Voice activity and pause detection."],
        ["Whisper large-v3 through Groq", "Groq transcription API", "Reference-guided hypothesis scoring for endings and tanween."],
        ["llama-3.3-70b-versatile through Groq", "Groq chat/completions API", "Automatic Arabic diacritizer for general text."],
    ]))

    body.append(heading("4. Architecture Overview"))
    body.append(paragraph("The architecture combines the original local Arabic speech pipeline with the proposed reference-guided architecture."))
    body.append(table([
        ["Input", "Processing", "Decision", "Output"],
        ["Reference Arabic text\nfixed exercise or custom user text", "If needed: Groq diacritizer adds harakat and final endings", "Extract target words with final haraka/tanween", "Expected endings per word"],
        ["Browser audio recording\nWebM/Opus", "Convert to 16 kHz mono WAV", "Run local wav2vec2 ensemble + Groq Whisper hypothesis scoring", "Observed ending evidence"],
        ["Reference + audio evidence", "Word matching, LLR, transcript ending parsing, tanween rules", "Signal fusion with confidence gates", "correct / incorrect / uncertain"],
        ["Full recording", "Silero VAD and delivery metrics", "Pause and speech-rate analysis", "Delivery feedback"],
    ]))

    body.append(heading("5. Pipeline Diagram"))
    diagram = (
        "[Browser UI]\n"
        "   |-- fixed exercise OR custom Arabic text\n"
        "   |-- microphone recording\n"
        "          |\n"
        "          v\n"
        "[FastAPI Backend]\n"
        "   |-- /api/diacritize: Groq LLaMA adds tashkeel when text is not vocalized\n"
        "   |-- /api/analyze: receives reference_text + audio\n"
        "          |\n"
        "          v\n"
        "[Audio Preparation]\n"
        "   WebM/Opus -> WAV -> 16 kHz mono -> normalized samples\n"
        "          |\n"
        "          v\n"
        "[Parallel Evidence Extraction]\n"
        "   1. wav2vec2 model 1 transcript + logits\n"
        "   2. wav2vec2 model 2 transcript + logits\n"
        "   3. wav2vec2 model 3 transcript + logits\n"
        "   4. Groq Whisper competing-hypothesis scores\n"
        "   5. Silero VAD delivery metrics\n"
        "          |\n"
        "          v\n"
        "[Arabic Ending Analyzer]\n"
        "   reference haraka extraction -> fuzzy word matching -> tanween rules -> word-aware LLR\n"
        "          |\n"
        "          v\n"
        "[Decision Fusion]\n"
        "   weighted evidence + confidence caps + uncertainty handling\n"
        "          |\n"
        "          v\n"
        "[Frontend Result]\n"
        "   word-by-word tashkeel/tanween findings + transcript + delivery metrics + debug details"
    )
    for line in diagram.splitlines():
        body.append(paragraph(line, mono=True))

    body.append(heading("6. Main Backend Flow"))
    body.append(table([
        ["Step", "Code area", "Description"],
        ["1", "frontend/static/js/app.js", "Records audio and sends FormData with exercise_id, reference_text, and audio file."],
        ["2", "backend/routers/analysis.py", "Receives request, converts audio, loads models, extracts expected final harakat from the reference."],
        ["3", "backend/utils/audio_utils.py", "Uses ffmpeg or bundled imageio-ffmpeg to convert WebM to WAV."],
        ["4", "backend/services/model_loader.py", "Loads local wav2vec2 models from Desktop\\etib_models so they are not re-downloaded each run."],
        ["5", "backend/services/acoustic_detector.py", "Runs ensemble voting, transcript matching, ending detection, tanween inference, and LLR scoring."],
        ["6", "Groq Whisper", "Scores multiple possible endings for each word using the same audio."],
        ["7", "combine_signals()", "Combines Whisper, wav2vec2 transcript evidence, and LLR into final status and confidence."],
        ["8", "frontend result cards", "Shows expected ending, detected ending, confidence, explanation, notes, transcript, and debug JSON."],
    ]))

    body.append(heading("7. Arabic Error Types Covered"))
    body.append(table([
        ["Error type", "Examples", "Current handling"],
        ["Final short vowels", "ـَ / ـُ / ـِ such as ذَهَبْتُ and مَكْتَبِ", "Detected through Whisper hypothesis scoring, LLR, and base-word fallback when ASR drops the vowel."],
        ["Tanween fatha", "كِتَابًا", "Detects final written ا/ان and scores competing tanween hypotheses."],
        ["Tanween damma", "طَالِبٌ", "Detects final ون when ASR writes it; missing bare base word can be marked as missing tanween."],
        ["Tanween kasra", "مَكْتَبٍ", "Detects final ين when ASR writes it; otherwise uses Whisper/LLR evidence."],
        ["Wrong long vowel ending", "مكتبِ pronounced as مكتبي, or damma as واو", "Transcript ending parser flags likely wrong long-vowel endings."],
        ["Reference text without harakat", "المدير ذهب إلى الاجتماع", "Groq diacritizer can create a vocalized reference; human review is still recommended for grammar-sensitive sentences."],
        ["Delivery issues", "Long pauses, slow rate, repeated words", "Silero/delivery module reports duration, speech rate, pauses, repetitions, and false starts when detected."],
    ]))

    body.append(heading("8. What Was Improved"))
    for item in [
        "Models are installed once on the Desktop instead of being downloaded every run.",
        "Added .env loading so API keys and configuration are read automatically by the backend.",
        "Added Groq status endpoint and frontend indicator.",
        "Added automatic Arabic diacritization endpoint for general text.",
        "Fixed audio conversion so Groq receives real WAV bytes instead of raw browser WebM mislabeled as WAV.",
        "Changed target selection to analyze all reference words with final tashkeel/tanween, not only manually chosen focus words.",
        "Added fuzzy Arabic word matching so shifted or slightly normalized ASR transcripts can still align to the reference word.",
        "Added expected-aware tanween interpretation and missing-tanween detection.",
        "Changed LLR from end-of-recording only to word-aware approximate frame regions.",
        "Added confidence caps and uncertainty states to avoid fake 100% certainty on weak acoustic evidence.",
    ]:
        body.append(bullet(item))

    body.append(heading("9. Accuracy Discussion"))
    body.append(paragraph(
        "The goal of >70% is realistic under the improved architecture when the reference text is fully diacritized and the recording is clear. "
        "The strongest path is exactly the one implemented: diacritized reference text, Groq Whisper hypothesis scoring, wav2vec2 ensemble evidence, and LLR backup."
    ))
    body.append(table([
        ["Condition", "Expected behavior"],
        ["Local wav2vec2 only", "Often 55-70%; short vowels and tanween may disappear from transcripts."],
        ["Diacritized reference + Groq Whisper scoring", "Likely above 70% on clean controlled recordings, with some cases around 80%."],
        ["Arbitrary unvocalized text without diacritizer", "Not reliable because the system does not know the correct final case ending."],
        ["Noisy audio or very short words", "Confidence decreases; system may mark uncertain."],
        ["Research-grade 85%+ robustness", "Would need a dedicated Arabic pronunciation dataset, forced alignment, or fine-tuning."],
    ]))

    body.append(heading("10. Current Problems and Limitations"))
    for item in [
        "Arabic ASR usually outputs letters without harakat, so a correct spoken damma/kasra/fatha may still appear as a bare word in transcript.",
        "Tanween is especially difficult because ASR may normalize it to bare words or long-vowel spellings.",
        "Word-level timing is approximate; a true forced aligner such as MFA would make LLR more reliable.",
        "Groq improves results but requires internet access and a valid API key.",
        "Automatic diacritization is not a grammar oracle. For academic evaluation, generated tashkeel should be reviewed for important examples.",
        "The current system is better for detecting final endings than for full phoneme-level Arabic pronunciation errors across the whole word.",
    ]:
        body.append(bullet(item))

    body.append(heading("11. How To Run"))
    body.append(paragraph("1. Models are already stored under:", mono=False))
    body.append(paragraph("C:\\Users\\Ali\\Desktop\\etib_models", mono=True))
    body.append(paragraph("2. Backend environment file exists in the project root:", mono=False))
    body.append(paragraph(".env with GROQ_API_KEY set", mono=True))
    body.append(paragraph("3. Start the backend from:", mono=False))
    body.append(paragraph("cd C:\\Users\\Ali\\Downloads\\etib_project\\etib\\backend", mono=True))
    body.append(paragraph("..\\venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000", mono=True))
    body.append(paragraph("4. Open the app:", mono=False))
    body.append(paragraph("http://127.0.0.1:8000", mono=True))

    body.append(heading("12. Recommended Demonstration Scenario"))
    for item in [
        "Use a fully diacritized sentence first, for example: ذَهَبْتُ إِلَى مَكْتَبِ الْمُدِيرِ.",
        "Record one correct reading.",
        "Record one reading with a deliberate tanween or final-vowel error.",
        "Show that the app returns word-level expected ending, detected ending, confidence, transcript, and explanation.",
        "Then paste unvocalized Arabic text and press the automatic diacritization button to demonstrate general-text support.",
    ]:
        body.append(bullet(item))

    body.append(heading("13. Conclusion"))
    body.append(paragraph(
        "The current module is no longer a simple ASR transcript checker. It is a hybrid Arabic pronunciation evaluator that uses reference-guided acoustic scoring, "
        "local model ensemble evidence, linguistic rules for final endings and tanween, and confidence-aware reporting. "
        "The architecture is appropriate for a student project targeting Arabic tashkeel/tanween feedback, with honest limitations documented for cases that need training or forced alignment."
    ))

    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}{sect}</w:body></w:document>"
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Arial"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr><w:pPr><w:spacing w:before="360" w:after="160"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:color w:val="2F75B5"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="A6A6A6"/><w:left w:val="single" w:sz="4" w:color="A6A6A6"/><w:bottom w:val="single" w:sz="4" w:color="A6A6A6"/><w:right w:val="single" w:sz="4" w:color="A6A6A6"/><w:insideH w:val="single" w:sz="4" w:color="D9D9D9"/><w:insideV w:val="single" w:sz="4" w:color="D9D9D9"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""


def main() -> None:
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        )
        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        )
        z.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>""",
        )
        z.writestr("word/document.xml", document_xml())
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/settings.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="720"/>
  <w:characterSpacingControl w:val="doNotCompress"/>
</w:settings>""")
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        z.writestr("docProps/core.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>ETIB Arabic Module Detailed Report</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""")
        z.writestr("docProps/app.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company>ETIB</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>""")
    HTML_OUT.write_text(build_html_report(), encoding="utf-8")
    print(OUT)
    print(HTML_OUT)


def build_html_report() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ETIB Arabic Module Detailed Report</title>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.55; max-width: 980px; margin: 32px auto; color: #1f2933; }
    h1 { color: #17365d; }
    h2 { color: #1f4e79; border-top: 1px solid #d6dde6; padding-top: 18px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }
    th, td { border: 1px solid #c9d3df; padding: 8px; vertical-align: top; }
    th { background: #ddebf7; }
    pre { background: #f6f8fa; padding: 14px; overflow-x: auto; border: 1px solid #d6dde6; }
    .download { display: inline-block; background: #1f4e79; color: white; padding: 12px 18px; border-radius: 6px; text-decoration: none; font-weight: bold; }
    .arabic { direction: rtl; font-family: Arial, sans-serif; font-size: 1.1em; }
  </style>
</head>
<body>
  <h1>ETIB Arabic Module: Pronunciation, Tashkeel and Tanween Error Detection</h1>
  <p><a class="download" href="ETIB_Arabic_Module_Detailed_Report.docx" download>Download the Word report (.docx)</a></p>
  <p><strong>Date:</strong> 30 June 2026</p>
  <p><strong>Scope:</strong> Arabic evaluation component only: final tashkeel, i'rab, tanween, and pronunciation feedback.</p>

  <h2>Architecture Summary</h2>
  <p>The system uses a hybrid reference-guided architecture: Groq LLaMA diacritizes arbitrary Arabic text, Groq Whisper scores competing pronunciation hypotheses, and three local wav2vec2 Arabic models plus word-aware LLR provide backup acoustic evidence.</p>

  <h2>Pipeline Diagram</h2>
  <pre>[Browser UI]
   |-- fixed exercise OR custom Arabic text
   |-- microphone recording
          |
          v
[FastAPI Backend]
   |-- /api/diacritize: Groq LLaMA adds tashkeel
   |-- /api/analyze: receives reference_text + audio
          |
          v
[Audio Preparation]
   WebM/Opus -> WAV -> 16 kHz mono
          |
          v
[Parallel Evidence Extraction]
   1. wav2vec2 model 1 transcript + logits
   2. wav2vec2 model 2 transcript + logits
   3. wav2vec2 model 3 transcript + logits
   4. Groq Whisper hypothesis scores
   5. Silero VAD delivery metrics
          |
          v
[Arabic Ending Analyzer]
   haraka extraction -> fuzzy matching -> tanween rules -> word-aware LLR
          |
          v
[Decision Fusion]
   weighted evidence + confidence caps + uncertainty handling
          |
          v
[Frontend Result]
   word-by-word findings + transcript + delivery metrics + debug details</pre>

  <h2>Models</h2>
  <table>
    <tr><th>Model</th><th>Location / API</th><th>Role</th></tr>
    <tr><td>jonatasgrosman/wav2vec2-large-xlsr-53-arabic</td><td>Desktop\\etib_models\\wav2vec2_jonatasgrosman_arabic</td><td>Arabic CTC ASR and logits</td></tr>
    <tr><td>elgeish/wav2vec2-large-xlsr-53-arabic</td><td>Desktop\\etib_models\\wav2vec2_elgeish_arabic</td><td>Arabic CTC ASR and logits</td></tr>
    <tr><td>KMFODA Arabic wav2vec2</td><td>Desktop\\etib_models\\wav2vec2_kmfoda_arabic</td><td>Arabic CTC ASR and logits</td></tr>
    <tr><td>Silero VAD</td><td>Desktop\\etib_models\\silero-vad</td><td>Voice activity and pauses</td></tr>
    <tr><td>Whisper large-v3 through Groq</td><td>Groq API</td><td>Reference-guided hypothesis scoring</td></tr>
    <tr><td>llama-3.3-70b-versatile through Groq</td><td>Groq API</td><td>Arabic diacritization</td></tr>
  </table>

  <h2>Problems and Limitations</h2>
  <ul>
    <li>Arabic ASR usually omits harakat, so short vowels are difficult to prove acoustically.</li>
    <li>Tanween may be normalized to bare words or long-vowel spellings.</li>
    <li>Word timing is approximate; a true forced aligner would improve LLR.</li>
    <li>Groq requires a valid API key and internet access.</li>
    <li>Automatic diacritization should be reviewed for important grammar examples.</li>
  </ul>

  <h2>Accuracy Claim</h2>
  <p>The realistic target is above 70% on clean, controlled recordings with fully diacritized reference text or Groq-assisted diacritization. A stable 80%+ on arbitrary Arabic text requires stronger alignment or training data.</p>
</body>
</html>
"""


if __name__ == "__main__":
    main()
