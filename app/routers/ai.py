from fastapi import APIRouter, Depends, HTTPException

from .. import ai_client, ai_mock, schemas
from ..deps import get_current_user_optional

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post("/chat", response_model=schemas.ChatResponse)
def chat(payload: schemas.ChatRequest, _user=Depends(get_current_user_optional)):
    # 로그인 없이도 튜터 챗봇 데모를 체험할 수 있도록 인증은 선택적으로만 확인한다.
    reply = ai_mock.generate_chat_reply(payload.message, payload.step)
    return schemas.ChatResponse(reply=reply)


@router.post("/analyze-reference")
def analyze_reference(payload: schemas.ReferenceAnalyzeRequest, _user=Depends(get_current_user_optional)):
    # 풀스코어 모드: 로그인 없이도 쓸 수 있게 인증은 선택적으로만 확인한다.
    if not ai_client.is_configured():
        raise HTTPException(501, "AI 분석을 쓰려면 서버에 GEMINI_API_KEY가 설정되어 있어야 해요.")
    try:
        return ai_mock.generate_reference_structure(payload.title, payload.artist, payload.composer, payload.duration)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"AI 분석에 실패했어요: {e}")
