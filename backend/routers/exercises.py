"""
exercises.py — GET /api/exercises, GET /api/exercises/{id}

Serves the 7 i'rab exercises from exercises.json.
"""

import os
import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger  = logging.getLogger(__name__)
router  = APIRouter()

EXERCISES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "exercises", "exercises.json"
)

_EXERCISES_CACHE: list[dict] = []


def _load_exercises() -> list[dict]:
    global _EXERCISES_CACHE
    if _EXERCISES_CACHE:
        return _EXERCISES_CACHE
    try:
        with open(EXERCISES_PATH, encoding="utf-8") as f:
            _EXERCISES_CACHE = json.load(f)
        logger.info(f"Loaded {len(_EXERCISES_CACHE)} exercises from {EXERCISES_PATH}")
    except FileNotFoundError:
        logger.warning(f"exercises.json not found at {EXERCISES_PATH} — using empty list")
        _EXERCISES_CACHE = []
    return _EXERCISES_CACHE


@router.get("/exercises")
async def list_exercises():
    """Return all exercises."""
    return JSONResponse(content=_load_exercises())


@router.get("/exercises/{exercise_id}")
async def get_exercise(exercise_id: str):
    """Return a single exercise by id."""
    exercises = _load_exercises()
    match = next((e for e in exercises if e["id"] == exercise_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return JSONResponse(content=match)
