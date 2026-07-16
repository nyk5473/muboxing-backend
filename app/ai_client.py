"""Gemini API 클라이언트. GEMINI_API_KEY가 없거나 호출이 실패하면
호출한 쪽에서 규칙 기반 mock으로 폴백할 수 있도록 예외를 그대로 던진다."""

import re
import time

import requests

from .config import settings

GEMINI_MODEL_CANDIDATES = ["gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]

# 무료 티어는 모델별 분당 요청 한도가 낮아서, 짧은 시간에 여러 번 호출하면 첫 번째
# 후보(gemini-flash-latest)조차 429를 받을 수 있다. 다음 후보로 넘어가기 전에 잠깐
# 기다렸다가 같은 모델로 재시도하면 대부분 곧바로 성공한다.
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_DEFAULT_WAIT = 4.0


def is_configured() -> bool:
    return bool(settings.gemini_api_key.strip())


def _parse_429(res: requests.Response) -> tuple[bool, float]:
    """(worth_retrying, wait_seconds)를 반환한다.

    무료 티어에서 하루 한도 자체가 0으로 막힌 모델은 몇 번을 재시도해도 계속 429가
    나므로, 일일 한도(PerDay) 위반이 포함돼 있으면 바로 포기하고 다음 후보로 넘어간다.
    분당 요청 한도(PerMinute)만 걸린 경우에만 잠깐 기다렸다가 재시도할 가치가 있다.
    """
    try:
        data = res.json()
        violations = data.get("error", {}).get("details", [])
        quota_ids = [
            v.get("quotaId", "")
            for d in violations if d.get("@type", "").endswith("QuotaFailure")
            for v in d.get("violations", [])
        ]
        if any("PerDay" in q for q in quota_ids):
            return False, 0.0
        for detail in violations:
            if detail.get("@type", "").endswith("RetryInfo"):
                m = re.match(r"([\d.]+)s?", detail.get("retryDelay", ""))
                if m:
                    return True, min(float(m.group(1)), 10.0)
        return True, RATE_LIMIT_DEFAULT_WAIT
    except Exception:  # noqa: BLE001
        return True, RATE_LIMIT_DEFAULT_WAIT


def call_gemini(system_prompt: str, contents: list[dict], max_tokens: int = 2048, temperature: float = 0.7) -> str:
    if not is_configured():
        raise RuntimeError("GEMINI_API_KEY가 설정되어 있지 않아요.")

    last_error = None
    for model in GEMINI_MODEL_CANDIDATES:
        attempts = RATE_LIMIT_RETRIES + 1
        for attempt in range(attempts):
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
                if res.status_code == 429 and attempt < attempts - 1:
                    # 분당 요청 한도(RPM)에만 걸린 경우엔 잠깐 기다렸다가 같은 모델로
                    # 재시도한다. 하루 한도 자체가 0인 모델은 재시도해도 의미가 없으므로
                    # 바로 다음 후보로 넘어간다.
                    worth_retrying, wait_seconds = _parse_429(res)
                    if worth_retrying:
                        time.sleep(wait_seconds)
                        continue
                if res.status_code in (400, 404, 429, 503):
                    last_error = f"{model}: HTTP {res.status_code} {res.text[:200]}"
                    break
                res.raise_for_status()
                data = res.json()
                candidate = data["candidates"][0]
                parts = candidate.get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                if not text:
                    last_error = f"{model}: 빈 응답 (finishReason={candidate.get('finishReason')})"
                    break
                return text
            except Exception as e:  # noqa: BLE001 - try next attempt/model candidate
                last_error = f"{model}: {e}"
                break

    raise RuntimeError(f"Gemini 호출에 실패했어요 ({last_error})")


def chat_once(system_prompt: str, user_message: str, max_tokens: int = 2048) -> str:
    return call_gemini(system_prompt, [{"role": "user", "parts": [{"text": user_message}]}], max_tokens=max_tokens)


def chat_with_history(system_prompt: str, history: list[dict], max_tokens: int = 1024) -> str:
    contents = [
        {"role": "model" if h["role"] == "assistant" else "user", "parts": [{"text": h["content"]}]}
        for h in history
    ]
    return call_gemini(system_prompt, contents, max_tokens=max_tokens, temperature=0.8)
