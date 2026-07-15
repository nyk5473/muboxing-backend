from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import ai_mock, models, schemas
from ..deps import get_current_user, get_db

router = APIRouter(prefix="/analyses", tags=["analyses"])


def _get_owned_session(analysis_id: int, db: Session, user: models.User) -> models.AnalysisSession:
    session = db.query(models.AnalysisSession).filter(models.AnalysisSession.id == analysis_id).first()
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "분석 세션을 찾을 수 없어요.")
    if session.owner_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "본인의 분석 세션만 볼 수 있어요.")
    return session


@router.post("", response_model=schemas.AnalysisDetail)
def create_analysis(
    payload: schemas.AnalysisCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    session = models.AnalysisSession(
        owner_id=user.id,
        title=payload.title,
        artist=payload.artist,
        genre=payload.genre,
        level=payload.level,
        youtube_url=payload.youtube_url,
        video_id=payload.video_id,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _to_detail(session)


@router.get("/mine", response_model=list[schemas.AnalysisSummary])
def list_mine(db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    sessions = (
        db.query(models.AnalysisSession)
        .filter(models.AnalysisSession.owner_id == user.id)
        .order_by(models.AnalysisSession.updated_at.desc())
        .all()
    )
    return [schemas.AnalysisSummary.from_orm_with_progress(s) for s in sessions]


@router.get("/{analysis_id}", response_model=schemas.AnalysisDetail)
def get_analysis(analysis_id: int, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    session = _get_owned_session(analysis_id, db, user)
    return _to_detail(session)


@router.patch("/{analysis_id}", response_model=schemas.AnalysisDetail)
def update_analysis(
    analysis_id: int,
    payload: schemas.AnalysisUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    session = _get_owned_session(analysis_id, db, user)
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(session, field, value)
    db.commit()
    db.refresh(session)
    return _to_detail(session)


@router.post("/{analysis_id}/complete", response_model=schemas.AnalysisDetail)
def complete_analysis(analysis_id: int, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    session = _get_owned_session(analysis_id, db, user)
    session.completed = True
    session.current_step = "feedback"
    db.commit()
    db.refresh(session)
    return _to_detail(session)


@router.post("/{analysis_id}/feedback", response_model=schemas.FeedbackOut)
def generate_feedback(analysis_id: int, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    session = _get_owned_session(analysis_id, db, user)
    feedback = ai_mock.generate_feedback(session)
    session.ai_feedback = feedback
    db.commit()
    return schemas.FeedbackOut(ai_feedback=feedback)


@router.post("/{analysis_id}/suggest-structure", response_model=schemas.SuggestStructureOut)
def suggest_structure(analysis_id: int, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    session = _get_owned_session(analysis_id, db, user)
    if not session.video_id and not session.youtube_url:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "먼저 Feeling 단계에서 유튜브 링크를 입력해주세요.")

    try:
        from ..dsp.structure_map import suggest_structure_from_youtube
    except ModuleNotFoundError as e:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"구조 자동 제안 기능에 필요한 패키지가 설치되어 있지 않아요 ({e.name}). "
            "requirements.txt의 librosa/yt-dlp를 설치해주세요.",
        )

    result = suggest_structure_from_youtube(session.video_id or session.youtube_url)

    session.sections_json = result["sections"]
    session.grid_json = result["grid"]
    db.commit()

    return schemas.SuggestStructureOut(**result)


def _to_detail(session: models.AnalysisSession) -> schemas.AnalysisDetail:
    progress = 100 if session.completed else schemas.STEP_PROGRESS.get(session.current_step, 20)
    return schemas.AnalysisDetail(
        id=session.id,
        title=session.title,
        artist=session.artist,
        genre=session.genre,
        level=session.level,
        youtube_url=session.youtube_url,
        video_id=session.video_id,
        current_step=session.current_step,
        progress=progress,
        note_beat=session.note_beat,
        note_melody=session.note_melody,
        note_structure=session.note_structure,
        note_apply=session.note_apply,
        beat_pattern_json=session.beat_pattern_json or [],
        melody_pattern_json=session.melody_pattern_json or {},
        sections_json=session.sections_json or [],
        grid_json=session.grid_json or {},
        ai_feedback=session.ai_feedback,
        completed=session.completed,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )
