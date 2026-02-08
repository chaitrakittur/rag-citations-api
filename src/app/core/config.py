from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_chat_model: str = "gpt-5.2"
    openai_embed_model: str = "text-embedding-3-small"

    top_k: int = 5
    min_context_chars: int = 400
    log_level: str = "INFO"


settings = Settings()
