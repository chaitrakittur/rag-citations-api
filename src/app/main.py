from fastapi import FastAPI
from app.core.config import settings
from app.core.logging_setup import setup_logger
from app.api.routes import router

logger = setup_logger(level=settings.log_level)

app = FastAPI(title="RAG Citations API", version="0.1.0")
app.include_router(router)

@app.on_event("startup")
def startup():
    logger.info("RAG Citations API starting | chat_model=%s | embed_model=%s", settings.openai_chat_model, settings.openai_embed_model)
