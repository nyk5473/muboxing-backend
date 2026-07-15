from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    analyses: Mapped[list["AnalysisSession"]] = relationship(back_populates="owner")
    posts: Mapped[list["CommunityPost"]] = relationship(back_populates="owner")


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    title: Mapped[str] = mapped_column(String(500), default="")
    artist: Mapped[str] = mapped_column(String(255), default="")
    genre: Mapped[str] = mapped_column(String(80), default="R&B / Soul")
    level: Mapped[str] = mapped_column(String(20), default="beginner")
    youtube_url: Mapped[str] = mapped_column(String(500), default="")
    video_id: Mapped[str] = mapped_column(String(20), default="")

    current_step: Mapped[str] = mapped_column(String(20), default="feeling")

    note_beat: Mapped[str] = mapped_column(Text, default="")
    note_melody: Mapped[str] = mapped_column(Text, default="")
    note_structure: Mapped[str] = mapped_column(Text, default="")
    note_apply: Mapped[str] = mapped_column(Text, default="")

    beat_pattern_json: Mapped[list] = mapped_column(JSON, default=list)
    melody_pattern_json: Mapped[dict] = mapped_column(JSON, default=dict)
    sections_json: Mapped[list] = mapped_column(JSON, default=list)
    grid_json: Mapped[dict] = mapped_column(JSON, default=dict)

    ai_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)

    owner: Mapped["User"] = relationship(back_populates="analyses")


class CommunityPost(Base):
    __tablename__ = "community_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    analysis_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_sessions.id"), nullable=True)

    song_title: Mapped[str] = mapped_column(String(255), default="")
    artist: Mapped[str] = mapped_column(String(255), default="")
    genre: Mapped[str] = mapped_column(String(80), default="")
    content: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    owner: Mapped["User"] = relationship(back_populates="posts")
    likes: Mapped[list["Like"]] = relationship(back_populates="post", cascade="all, delete-orphan")
    comments: Mapped[list["Comment"]] = relationship(back_populates="post", cascade="all, delete-orphan")


class Like(Base):
    __tablename__ = "likes"
    __table_args__ = (UniqueConstraint("post_id", "user_id", name="uq_like_post_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("community_posts.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    post: Mapped["CommunityPost"] = relationship(back_populates="likes")


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("community_posts.id"), nullable=False)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    post: Mapped["CommunityPost"] = relationship(back_populates="comments")
    owner: Mapped["User"] = relationship()
