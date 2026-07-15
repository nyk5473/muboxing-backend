from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 60 * 24 * 7
    database_url: str = "sqlite:///./muboxing.db"
    cors_origins: str = "*"
    algorithm: str = "HS256"
    gemini_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
