"""
Arabic diacritization helper.

Uses Groq LLaMA when GROQ_API_KEY is configured. This is intended to create
the expected reference tashkeel for arbitrary Arabic text before pronunciation
analysis. Human review is still recommended for high-stakes grammar.
"""

import os
import logging

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DIACRITIZER_MODEL = "llama-3.3-70b-versatile"


async def diacritize_arabic_text(text: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {
            "error": "GROQ_API_KEY not configured",
            "diacritized_text": text,
            "notes": "Add GROQ_API_KEY to .env, restart the server, then use automatic diacritization.",
        }

    prompt = f"""شكّل النص العربي التالي تشكيلاً كاملاً مناسبًا للقراءة الجهرية.

القواعد:
- أعد النص العربي فقط، بلا شرح.
- حافظ على الكلمات والترقيم قدر الإمكان.
- ركّز خصوصًا على الحركة الأخيرة لكل كلمة والتنوين.
- إذا كان هناك احتمالان نحويان، اختر الأرجح سياقيًا.

النص:
{text}"""

    try:
        import httpx

        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": DIACRITIZER_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "أنت خبير في تشكيل العربية الفصحى. أعد النص مشكولًا فقط.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1200,
                },
            )

        if resp.status_code != 200:
            logger.warning("Groq diacritizer error %s: %s", resp.status_code, resp.text[:300])
            return {"error": f"Groq API error {resp.status_code}", "diacritized_text": text}

        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = content.strip("`").strip()
        if content.startswith("النص:"):
            content = content.split(":", 1)[1].strip()
        return {
            "diacritized_text": content,
            "model": DIACRITIZER_MODEL,
            "notes": "Review the generated tashkeel before recording if grammar matters.",
        }
    except Exception as e:
        logger.exception("Diacritization failed")
        return {"error": str(e), "diacritized_text": text}
