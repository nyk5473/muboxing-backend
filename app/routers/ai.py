from fastapi import APIRouter, Depends

from .. import ai_mock, schemas
from ..deps import get_current_user_optional

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post("/chat", response_model=schemas.ChatResponse)
def chat(payload: schemas.ChatRequest, _user=Depends(get_current_user_optional)):
    # 로그인 없이도 튜터 챗봇 데모를 체험할 수 있도록 인증은 선택적으로만 확인한다.
    reply = ai_mock.generate_chat_reply(payload.message, payload.step)
    return schemas.ChatResponse(reply=reply)
