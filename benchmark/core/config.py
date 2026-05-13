from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    default_language: str = "zh-CN"
    data_dir: str = "./data"
    corpus_jsonl_path: str = "./data/corpus.jsonl"
    samples_jsonl_path: str = "./data/samples.jsonl"
    items_jsonl_path: str = "./data/items.jsonl"

    llm_enabled: bool = True
    llm_provider: str = "openai_compatible"
    llm_model: str = "qwen3-32b"
    llm_api_key: str = "EMPTY"
    llm_api_base: str = "http://10.208.62.156:8002/v1"
    llm_timeout_seconds: float = 45.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_request_retries: int = 1

    judge_provider: str = "same_as_llm"
    judge_model: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
