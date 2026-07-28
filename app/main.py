import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models
from .config import settings
from .database import Base, engine
from .routers import ai, analyses, auth, community

Base.metadata.create_all(bind=engine)

app = FastAPI(title="뮤박싱(MuBoxing) 백엔드")

origins = ["*"] if settings.cors_origins.strip() == "*" else [
    o.strip() for o in settings.cors_origins.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(analyses.router)
app.include_router(community.router)
app.include_router(ai.router)


def _warmup_dsp():
    # librosa/numba는 프로세스에서 처음 호출될 때 JIT 컴파일 때문에 수십~100초 넘게
    # 걸릴 수 있다. 콜드 스타트 직후 사용자의 첫 분석 요청이 그 비용을 그대로 떠안지
    # 않도록, 서버 기동 시 백그라운드에서 짧은 더미 신호로 미리 한 번 실행해둔다.
    try:
        import numpy as np

        from .dsp.analysis import analyze_audio

        sr = 11025
        duration = 6
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        analyze_audio(y, sr)
    except Exception:  # noqa: BLE001 - 워밍업 실패는 무시, 실제 요청에서 다시 시도된다
        pass


@app.on_event("startup")
def warmup_on_startup():
    threading.Thread(target=_warmup_dsp, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok"}
