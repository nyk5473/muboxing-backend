import logging
import os
import tempfile
from contextlib import suppress

import librosa
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from .. import ai_client, ai_mock, schemas
from ..deps import get_current_user_optional

router = APIRouter(prefix="/ai", tags=["ai"])
logger = logging.getLogger(__name__)

SR = 22050
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _convert_dsp_result(result: dict) -> dict:
    return {
        "data": {
            "TOTAL": round(result["duration"]),
            "sections": result["sections"],
            "tracks": result["tracks"],
        },
        "harmony": result["harmony"],
        "quiz": result.get("quiz", []),
    }


@router.post("/chat", response_model=schemas.ChatResponse)
def chat(payload: schemas.ChatRequest, _user=Depends(get_current_user_optional)):
    # 로그인 없이도 튜터 챗봇 데모를 체험할 수 있도록 인증은 선택적으로만 확인한다.
    reply = ai_mock.generate_chat_reply(payload.message, payload.step)
    return schemas.ChatResponse(reply=reply)


@router.post("/analyze-reference")
def analyze_reference(payload: schemas.ReferenceAnalyzeRequest, _user=Depends(get_current_user_optional)):
    # 풀스코어 모드: 로그인 없이도 쓸 수 있게 인증은 선택적으로만 확인한다.
    # video_id가 있으면 실제 오디오를 내려받아 librosa로 분석하고, 실패하면 곡명 기반
    # Gemini 추정으로 폴백한다 — 재생바/구간이 실제 곡과 어긋나지 않게 하기 위함.
    if payload.video_id:
        try:
            from ..dsp.analysis import analyze_audio
            from ..dsp.structure_map import _download_and_load
            y, sr = _download_and_load(payload.video_id)
            result = analyze_audio(y, sr)
            return _convert_dsp_result(result)
        except ModuleNotFoundError as e:
            logger.warning("DSP 패키지 미설치, 메타데이터 기반 분석으로 폴백: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("실제 오디오 분석 실패, 메타데이터 기반 분석으로 폴백: %s", e)

    if not ai_client.is_configured():
        raise HTTPException(501, "AI 분석을 쓰려면 서버에 GEMINI_API_KEY가 설정되어 있어야 해요.")
    try:
        return ai_mock.generate_reference_structure(payload.title, payload.artist, payload.composer, payload.duration)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"AI 분석에 실패했어요: {e}")


@router.post("/analyze-reference-upload")
async def analyze_reference_upload(file: UploadFile = File(...), _user=Depends(get_current_user_optional)):
    # 풀스코어 모드에서 로컬 오디오 파일을 업로드했을 때, 실제 파형을 librosa로 분석한다.
    from ..dsp.analysis import analyze_audio

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "파일이 50MB를 초과해요.")

    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)

        try:
            y, sr = librosa.load(tmp_path, sr=SR, mono=True)
        except Exception as e:
            raise HTTPException(400, f"오디오를 읽을 수 없어요: {e}")

        if len(y) == 0:
            raise HTTPException(400, "오디오를 읽을 수 없어요.")

        try:
            result = analyze_audio(y, sr)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"분석 중 오류가 발생했어요: {e}")

        return _convert_dsp_result(result)
    finally:
        with suppress(OSError):
            os.remove(tmp_path)
