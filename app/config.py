from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 60 * 24 * 7
    database_url: str = "sqlite:///./muboxing.db"
    cors_origins: str = "*"
    algorithm: str = "HS256"
    gemini_api_key: str = ""
    # 유튜브가 클라우드 서버 IP를 봇으로 의심해 오디오 다운로드를 막는 문제 대응용.
    # 로그인된 브라우저에서 내보낸 cookies.txt(Netscape 형식) 내용을 그대로 넣으면
    # yt-dlp가 로그인된 요청처럼 보내서 차단을 우회할 수 있다.
    ytdlp_cookies: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
