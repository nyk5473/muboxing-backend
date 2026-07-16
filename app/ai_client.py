"""Gemini API 클라이언트. GEMINI_API_KEY가 없거나 호출이 실패하면
호출한 쪽에서 규칙 기반 mock으로 폴백할 수 있도록 예외를 그대로 던진다."""

import requests

from .config import settings

GEMINI_MODEL_CANDIDATES = ["gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]


def is_configured() -> bool:
    return bool(settings.gemini_api_key.strip())


def call_gemini(system_prompt: str, contents: list[dict], max_tokens: int = 2048, temperature: float = 0.7) -> str:
    if not is_configured():
        raise RuntimeError("GEMINI_API_KEY가 설정되어 있지 않아요.")

    last_error = None
    for model in GEMINI_MODEL_CANDIDATES:
        try:
            res = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": settings.gemini_api_key},
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature,
                        # 2.5 계열의 "thinking" 모드가 maxOutputTokens를 내부 추론에 먼저 소모해서
                        # 실제 답변이 잘리는 걸 막기 위해 thinking을 꺼둔다. 미지원 모델이면 400을
                        # 받고 다음 후보로 넘어간다.
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
                timeout=25,
            )
            if res.status_code in (400, 404, 429, 503):
                last_error = f"{model}: HTTP {res.status_code} {res.text[:200]}"
                continue
            res.raise_for_status()
            data = res.json()
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                last_error = f"{model}: 빈 응답 (finishReason={candidate.get('finishReason')})"
                continue
            return text
        except Exception as e:  # noqa: BLE001 - try next model candidate
            last_error = f"{model}: {e}"
            continue

    raise RuntimeError(f"Gemini 호출에 실패했어요 ({last_error})")


def chat_once(system_prompt: str, user_message: str, max_tokens: int = 2048) -> str:
    return call_gemini(system_prompt, [{"role": "user", "parts": [{"text": user_message}]}], max_tokens=max_tokens)


def chat_with_history(system_prompt: str, history: list[dict], max_tokens: int = 1024) -> str:
    contents = [
        {"role": "model" if h["role"] == "assistant" else "user", "parts": [{"text": h["content"]}]}
        for h in history
    ]
    return call_gemini(system_prompt, contents, max_tokens=max_tokens, temperature=0.8)
