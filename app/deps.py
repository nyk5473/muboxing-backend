from typing import Generator, Optional

from fastapi import Depends, HTTPException, Request, status

from . import models
from .database import SessionLocal
from .security import decode_access_token


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _extract_token(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


def get_current_user(request: Request, db=Depends(get_db)) -> models.User:
    token = _extract_token(request)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "로그인이 필요해요.")
    email = decode_access_token(token)
    if not email:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "토큰이 유효하지 않아요.")
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "사용자를 찾을 수 없어요.")
    return user


def get_current_user_optional(request: Request, db=Depends(get_db)) -> Optional[models.User]:
    token = _extract_token(request)
    if not token:
        return None
    email = decode_access_token(token)
    if not email:
        return None
    return db.query(models.User).filter(models.User.email == email).first()
