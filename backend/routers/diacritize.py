"""Arabic diacritization endpoints."""

import os
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.diacritizer import diacritize_arabic_text

router = APIRouter()


class DiacritizeRequest(BaseModel):
    text: str


@router.get("/groq-status")
async def groq_status():
    return {"configured": bool(os.environ.get("GROQ_API_KEY"))}


@router.post("/diacritize")
async def diacritize_endpoint(req: DiacritizeRequest):
    result = await diacritize_arabic_text(req.text)
    return JSONResponse(content=result)
