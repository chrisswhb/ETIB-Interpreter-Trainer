"""
translation_eval.py — POST /api/evaluate-translation

Layer 1: LLM-based translation quality evaluation (Dr's main addition).
Accepts source text + student translation and returns the layer1_translation JSON.
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from services.translation_evaluator import evaluate_translation

logger = logging.getLogger(__name__)
router = APIRouter()


class TranslationEvalRequest(BaseModel):
    source_language: str = "en"          # fr | en
    target_language: str = "ar"          # ar | en | fr
    feedback_language: str = "ar"        # language for human-readable feedback
    source_text: str
    student_translation: str
    reference_materials: Optional[list]  = None   # RAG-retrieved entries
    reference_translations: Optional[str] = None  # optional gold reference(s)
    known_interference_matches: Optional[list] = None  # from wrong-form matcher


@router.post("/evaluate-translation")
async def evaluate_translation_endpoint(req: TranslationEvalRequest):
    """
    Evaluate a student's translation against its source.
    Returns layer1_translation JSON object per Dr's rubric schema.
    """
    result = await evaluate_translation(
        source_language=req.source_language,
        target_language=req.target_language,
        feedback_language=req.feedback_language,
        source_text=req.source_text,
        student_translation=req.student_translation,
        reference_materials=req.reference_materials,
        reference_translations=req.reference_translations,
        known_interference_matches=req.known_interference_matches,
    )
    return JSONResponse(content=result)
