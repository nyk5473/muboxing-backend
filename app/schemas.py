from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

STEP_PROGRESS = {
    "feeling": 20,
    "beat": 40,
    "melody": 60,
    "structure": 80,
    "feedback": 100,
}


# ---------- Auth ----------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    nickname: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    nickname: str

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------- Analysis Sessions ----------
class AnalysisCreate(BaseModel):
    title: str = ""
    artist: str = ""
    genre: str = "R&B / Soul"
    level: str = "beginner"
    youtube_url: str = ""
    video_id: str = ""


class AnalysisUpdate(BaseModel):
    title: str | None = None
    artist: str | None = None
    genre: str | None = None
    level: str | None = None
    youtube_url: str | None = None
    video_id: str | None = None
    current_step: str | None = None
    note_beat: str | None = None
    note_melody: str | None = None
    note_structure: str | None = None
    note_apply: str | None = None
    beat_pattern_json: list | None = None
    melody_pattern_json: dict | None = None
    sections_json: list | None = None
    grid_json: dict | None = None


class AnalysisSummary(BaseModel):
    id: int
    title: str
    artist: str
    genre: str
    current_step: str
    progress: int
    completed: bool
    updated_at: datetime

    class Config:
        from_attributes = True

    @staticmethod
    def from_orm_with_progress(obj):
        progress = 100 if obj.completed else STEP_PROGRESS.get(obj.current_step, 20)
        return AnalysisSummary(
            id=obj.id,
            title=obj.title,
            artist=obj.artist,
            genre=obj.genre,
            current_step=obj.current_step,
            progress=progress,
            completed=obj.completed,
            updated_at=obj.updated_at,
        )


class AnalysisDetail(BaseModel):
    id: int
    title: str
    artist: str
    genre: str
    level: str
    youtube_url: str
    video_id: str
    current_step: str
    progress: int
    note_beat: str
    note_melody: str
    note_structure: str
    note_apply: str
    beat_pattern_json: list
    melody_pattern_json: dict
    sections_json: list
    grid_json: dict
    ai_feedback: str | None
    completed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ---------- AI ----------
class FeedbackOut(BaseModel):
    ai_feedback: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    step: str | None = None
    history: list[ChatMessage] = []


class ReferenceAnalyzeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    artist: str = ""
    composer: str = ""
    duration: int = Field(default=210, ge=10, le=1200)


class ChatResponse(BaseModel):
    reply: str


class SuggestStructureOut(BaseModel):
    sections: list[str]
    grid: dict[str, bool]
    note: str


# ---------- Community ----------
class CommunityPostCreate(BaseModel):
    analysis_id: int | None = None
    song_title: str = Field(min_length=1, max_length=255)
    artist: str = ""
    genre: str = ""
    content: str = Field(min_length=1)


class CommunityPostOut(BaseModel):
    id: int
    owner_nickname: str
    analysis_id: int | None
    song_title: str
    artist: str
    genre: str
    content: str
    likes_count: int
    comments_count: int
    liked_by_me: bool
    created_at: datetime


class CommentCreate(BaseModel):
    content: str = Field(min_length=1)


class CommentOut(BaseModel):
    id: int
    owner_nickname: str
    content: str
    created_at: datetime
