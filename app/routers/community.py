from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import models, schemas
from ..deps import get_current_user, get_current_user_optional, get_db

router = APIRouter(prefix="/community", tags=["community"])


def _to_post_out(post: models.CommunityPost, current_user: models.User | None) -> schemas.CommunityPostOut:
    liked_by_me = bool(
        current_user and any(like.user_id == current_user.id for like in post.likes)
    )
    return schemas.CommunityPostOut(
        id=post.id,
        owner_nickname=post.owner.nickname,
        analysis_id=post.analysis_id,
        song_title=post.song_title,
        artist=post.artist,
        genre=post.genre,
        content=post.content,
        likes_count=len(post.likes),
        comments_count=len(post.comments),
        liked_by_me=liked_by_me,
        created_at=post.created_at,
    )


@router.get("/posts", response_model=list[schemas.CommunityPostOut])
def list_posts(
    genre: str | None = None,
    limit: int = 30,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: models.User | None = Depends(get_current_user_optional),
):
    query = db.query(models.CommunityPost)
    if genre and genre != "전체":
        query = query.filter(models.CommunityPost.genre == genre)
    posts = (
        query.order_by(models.CommunityPost.created_at.desc())
        .offset(offset)
        .limit(min(limit, 100))
        .all()
    )
    return [_to_post_out(p, current_user) for p in posts]


@router.post("/posts", response_model=schemas.CommunityPostOut)
def create_post(
    payload: schemas.CommunityPostCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if payload.analysis_id is not None:
        owned = (
            db.query(models.AnalysisSession)
            .filter(
                models.AnalysisSession.id == payload.analysis_id,
                models.AnalysisSession.owner_id == current_user.id,
            )
            .first()
        )
        if not owned:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "연결하려는 분석 세션을 찾을 수 없어요.")

    post = models.CommunityPost(
        owner_id=current_user.id,
        analysis_id=payload.analysis_id,
        song_title=payload.song_title,
        artist=payload.artist,
        genre=payload.genre,
        content=payload.content,
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return _to_post_out(post, current_user)


@router.post("/posts/{post_id}/like")
def toggle_like(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    post = db.query(models.CommunityPost).filter(models.CommunityPost.id == post_id).first()
    if not post:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "게시글을 찾을 수 없어요.")

    existing = (
        db.query(models.Like)
        .filter(models.Like.post_id == post_id, models.Like.user_id == current_user.id)
        .first()
    )
    if existing:
        db.delete(existing)
        liked = False
    else:
        db.add(models.Like(post_id=post_id, user_id=current_user.id))
        liked = True
    db.commit()
    db.refresh(post)
    return {"liked": liked, "likes_count": len(post.likes)}


@router.get("/posts/{post_id}/comments", response_model=list[schemas.CommentOut])
def list_comments(post_id: int, db: Session = Depends(get_db)):
    post = db.query(models.CommunityPost).filter(models.CommunityPost.id == post_id).first()
    if not post:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "게시글을 찾을 수 없어요.")
    return [
        schemas.CommentOut(id=c.id, owner_nickname=c.owner.nickname, content=c.content, created_at=c.created_at)
        for c in sorted(post.comments, key=lambda c: c.created_at)
    ]


@router.post("/posts/{post_id}/comments", response_model=schemas.CommentOut)
def create_comment(
    post_id: int,
    payload: schemas.CommentCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    post = db.query(models.CommunityPost).filter(models.CommunityPost.id == post_id).first()
    if not post:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "게시글을 찾을 수 없어요.")

    comment = models.Comment(post_id=post_id, owner_id=current_user.id, content=payload.content)
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return schemas.CommentOut(
        id=comment.id, owner_nickname=current_user.nickname, content=comment.content, created_at=comment.created_at
    )
