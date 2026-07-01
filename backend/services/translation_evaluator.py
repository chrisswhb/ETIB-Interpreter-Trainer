"""
translation_evaluator.py — Layer 1: LLM-based translation quality evaluation.

Dr's spec: evaluate student translation against source using Groq LLaMA-3.3-70B
(or any capable LLM) with the 6-dimension rubric prompt.

Dimensions:
  1. fidelity       — meaning preservation
  2. terminology    — correct equivalents (checked against reference glossary)
  3. interference   — calque / source-structure leakage (PRIORITY for Arabic)
  4. grammar        — agreement, case, prepositions
  5. orthography    — Arabic only (hamza, alif layyna, tā' marbūṭa, punctuation)
  6. register       — formality appropriate to text type

Output: layer1_translation JSON object per Dr's schema
"""

import os
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
EVAL_MODEL    = "llama-3.3-70b-versatile"   # Dr: "keep your current Groq LLaMA-3.3-70B"


LAYER1_SYSTEM_PROMPT = """You are a senior translation reviewer for the École de Traduction et d'Interprétation de Beyrouth (ETIB). You evaluate a student's translation against its source and produce precise, teachable, span-level feedback. You are rigorous but fair: a passage has valid renderings, so you separate genuine errors from optional stylistic suggestions; you never penalise legitimate variation.

RUBRIC — evaluate every dimension (dimension 5 applies only when target_language = ar):
1. fidelity      — meaning preserved; no omission, addition, or distortion
2. terminology   — correct, consistent equivalents; check against reference materials
3. interference  — calque and source-structure leakage that is wrong or unidiomatic in the target; literal preposition/word transfers, foreign word order, stacked-muḍāf constructions. THIS IS THE PRIORITY DIMENSION FOR ARABIC TARGETS.
4. grammar       — agreement, case/government logic, correct prepositions (ḥurūf al-jarr), tense/aspect, definiteness
5. orthography   — Arabic only: hamza, alif layyina/maqṣūra, tā' marbūṭa, spelling, punctuation (ʿalāmāt al-tarqīm)
6. register      — formality and domain style appropriate to the text type

METHOD:
- Align source and target into sentence-level segments; evaluate segment by segment
- For each problem, emit exactly one finding with: exact target span, dimension, severity (error|warning|suggestion), corrected version, one-to-two-sentence explanation, and citations
- Treat every entry in known_interference_matches as a strong prior: confirm or override on evidence; when confirmed, cite its rule_id
- Report each distinct issue once; do not repeat the same issue across segments

CITATIONS (strict):
- Cite ONLY entries present in reference_materials, by their "id"
- If a finding is genuinely correct but no entry supports it, set evidence_type = "model_judgment" and leave citations empty
- NEVER invent a rule, source, page/locator, or quotation

OUTPUT:
Return ONLY one valid JSON object. No text before or after the JSON. Use the exact enum values. Use feedback_language for all human-readable fields and the target language for all "correction" fields."""


def build_layer1_prompt(
    source_language: str,
    target_language: str,
    feedback_language: str,
    source_text: str,
    student_translation: str,
    reference_materials: list = None,
    reference_translations: str = "",
    known_interference_matches: list = None,
) -> str:
    ref_mat    = json.dumps(reference_materials or [], ensure_ascii=False)
    known_int  = json.dumps(known_interference_matches or [], ensure_ascii=False)
    ref_trans  = reference_translations or ""

    return f"""INPUTS
- source_language: {source_language}
- target_language: {target_language}
- feedback_language: {feedback_language}
- source_text:
{source_text}
- student_translation:
{student_translation}
- reference_materials: (your ONLY citable authorities)
{ref_mat}
- reference_translations: (optional; for comparison only, NOT the single correct answer)
{ref_trans}
- known_interference_matches: (high-precision pre-matches from wrong-form matcher)
{known_int}

OUTPUT SCHEMA:
{{
  "overall_summary": "string — 1-3 sentences in feedback_language",
  "segments": [
    {{
      "segment_id": "string",
      "source_segment": "string",
      "target_segment": "string",
      "findings": [
        {{
          "finding_id": "string",
          "dimension": "fidelity | terminology | interference | grammar | orthography | register",
          "severity": "error | warning | suggestion",
          "target_span": "string — exact substring of target_segment",
          "char_start": null,
          "char_end": null,
          "issue": "string — short label",
          "explanation": "string — 1-2 sentences in {feedback_language}",
          "correction": "string — suggested target-language text",
          "evidence_type": "reference_backed | model_judgment",
          "citations": [{{ "source_id": "string", "rule_id": "string", "locator": "string", "quote": null }}]
        }}
      ]
    }}
  ],
  "dimension_summary": {{
    "fidelity": "string | band 0-3",
    "terminology": "string | band 0-3",
    "interference": "string | band 0-3",
    "grammar": "string | band 0-3",
    "orthography": "string | band 0-3 (null if target != ar)",
    "register": "string | band 0-3"
  }},
  "notes": "string"
}}"""


async def evaluate_translation(
    source_language: str,
    target_language: str,
    feedback_language: str,
    source_text: str,
    student_translation: str,
    reference_materials: Optional[list] = None,
    reference_translations: Optional[str] = None,
    known_interference_matches: Optional[list] = None,
) -> dict:
    """
    Call Groq LLaMA to evaluate a student translation and return
    the layer1_translation JSON object.
    """
    if not GROQ_API_KEY:
        return {
            "error": "GROQ_API_KEY not configured",
            "overall_summary": "Translation evaluation unavailable — API key missing.",
            "segments": [],
            "dimension_summary": {},
            "notes": "",
        }

    user_prompt = build_layer1_prompt(
        source_language=source_language,
        target_language=target_language,
        feedback_language=feedback_language,
        source_text=source_text,
        student_translation=student_translation,
        reference_materials=reference_materials,
        reference_translations=reference_translations or "",
        known_interference_matches=known_interference_matches,
    )

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EVAL_MODEL,
                    "messages": [
                        {"role": "system", "content": LAYER1_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
            )

        if resp.status_code != 200:
            logger.error(f"Groq LLM error {resp.status_code}: {resp.text[:300]}")
            return {"error": f"LLM API error {resp.status_code}"}

        raw = resp.json()["choices"][0]["message"]["content"]
        # Strip any accidental markdown code fences
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from LLM: {e}")
        return {"error": "Could not parse LLM response as JSON", "raw": raw[:500]}
    except Exception as e:
        logger.error(f"Translation evaluation failed: {e}")
        return {"error": str(e)}
