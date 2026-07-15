# MuBoxing 백엔드

FastAPI + SQLite 기반 백엔드. `html/거의 최종.html` 프론트엔드와 짝을 이룹니다.

## 실행 방법

```bash
cd muboxing-backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # 필요시 SECRET_KEY 등 수정
uvicorn app.main:app --reload --port 8000
```

- API 문서: http://127.0.0.1:8000/docs
- DB는 `muboxing.db` (SQLite) 파일로 자동 생성됩니다.

### librosa / yt-dlp (구조 자동 제안 기능)

`requirements.txt`에 포함돼 있지만, 실제로 유튜브 오디오를 다운로드하려면 시스템에 **ffmpeg**가 설치되어 있어야 합니다.
설치돼 있지 않아도 나머지 기능(회원가입, 분석 세션 저장, AI 피드백/챗봇, 커뮤니티)은 정상 동작하고,
"AI가 구조 초안 제안" 버튼만 에러 메시지를 반환합니다.

윈도우에서 ffmpeg가 없다면: `winget install Gyan.FFmpeg` (설치 후 새 터미널에서 `ffmpeg -version`으로 확인)

### GEMINI_API_KEY (실제 AI 피드백/챗봇)

1. https://aistudio.google.com/apikey 에서 무료로 키를 발급받으세요.
2. `.env` 파일의 `GEMINI_API_KEY=` 뒤에 붙여넣고 서버를 재시작하세요.
3. 키가 없거나 호출이 실패하면(네트워크 오류, 쿼터 초과, 잘못된 키 등) 자동으로 규칙 기반 mock 응답으로
   폴백하므로, 키를 넣지 않아도 앱은 항상 정상 동작합니다.

## 프론트엔드 연동

`html/거의 최종.html`은 `API_BASE = 'http://127.0.0.1:8000'`로 하드코딩돼 있습니다. 배포 환경이 바뀌면
파일 상단 script 블록의 `API_BASE` 값만 바꾸면 됩니다.

프론트엔드는 정적 파일이라 `file://`로 직접 열어도 대부분 동작하지만, 브라우저 정책상
`python -m http.server`처럼 로컬 웹서버로 여는 걸 권장합니다.

## AI 피드백/챗봇 동작 방식

`app/ai_client.py`가 실제 Gemini API 호출을 담당하고, `app/ai_mock.py`의 `generate_feedback()` /
`generate_chat_reply()`가 그 호출을 감싸서 실패 시 규칙 기반 mock으로 자동 폴백합니다.
GEMINI_API_KEY를 설정하면 실제 AI 응답을, 설정하지 않으면 기존 mock 응답을 그대로 받게 됩니다.
