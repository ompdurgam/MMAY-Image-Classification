"""
main.py – Entry point only. No logic lives here.
"""
import uvicorn

from app.application import create_app
from app.core.config import get_settings

app = create_app()

if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host      = cfg.APP_HOST,
        port      = cfg.APP_PORT,
        log_level = cfg.LOG_LEVEL,
        reload    = True,
    )