"""260710/backend의 librosa+yt-dlp DSP 분석 결과를
거의 최종.html의 DAW 그리드 포맷(SECTIONS 배열 + state.grid 딕셔너리)으로 변환한다.

거의 최종.html의 TRACKS 순서 (index 고정):
  0: beat (드럼)   1: bass   2: chords (신스/건반)   3: lead (리드 악기)   4: vocal (보컬/멜로디)

DSP 엔진의 트랙 순서: Drums, Bass, Chords, Vocal, FX
FX ↔ lead는 완벽히 대응되지 않는 근사치 매핑임 (Structure 초안 제안용).
"""

import os
import re
import tempfile
from contextlib import suppress

import librosa
import yt_dlp
from fastapi import HTTPException

from .analysis import analyze_audio

SR = 22050

DSP_TRACK_TO_FRONTEND_INDEX = {
    "Drums": 0,   # beat
    "Bass": 1,    # bass
    "Chords": 2,  # chords
    "FX": 3,      # lead (근사치)
    "Vocal": 4,   # vocal
}

_VIDEO_ID_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"^([a-zA-Z0-9_-]{11})$"),
]


def extract_video_id(text: str) -> str | None:
    text = (text or "").strip()
    for pattern in _VIDEO_ID_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def _download_and_load(video_id: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "retries": 2,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        except yt_dlp.utils.DownloadError as e:
            raise HTTPException(502, detail=f"유튜브 오디오를 가져오지 못했어요: {e}")

        downloaded = [f for f in os.listdir(tmpdir) if f.startswith("audio.")]
        if not downloaded:
            raise HTTPException(502, detail="오디오 파일을 찾지 못했어요.")
        audio_path = os.path.join(tmpdir, downloaded[0])

        try:
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
        except Exception as e:
            raise HTTPException(500, detail=f"오디오 디코딩에 실패했어요: {e}")

    if len(y) == 0:
        raise HTTPException(400, detail="오디오를 읽을 수 없어요.")
    return y, sr


def suggest_structure_from_youtube(video_id_or_url: str) -> dict:
    video_id = extract_video_id(video_id_or_url)
    if not video_id:
        raise HTTPException(400, detail="유효한 유튜브 링크/videoId가 아니에요.")

    y, sr = _download_and_load(video_id)

    try:
        result = analyze_audio(y, sr)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"분석 중 오류가 발생했어요: {e}")

    sections = [s["label"] for s in result["sections"]]

    grid: dict[str, bool] = {}
    for track in result["tracks"]:
        track_idx = DSP_TRACK_TO_FRONTEND_INDEX.get(track["name"])
        if track_idx is None:
            continue
        for i, seg in enumerate(track["segs"]):
            if i >= len(sections):
                break
            if not seg.get("ghost"):
                grid[f"{track_idx}-{sections[i]}"] = True

    return {
        "sections": sections,
        "grid": grid,
        "note": result["harmony"]["noteText"],
    }
