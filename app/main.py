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


@app.get("/health")
def health():
    return {"status": "ok"}
