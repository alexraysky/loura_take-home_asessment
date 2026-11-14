"""FastAPI app factory, lifespan and error handlers."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import ApiError, ErrorBody, ErrorEnvelope, router
from app.clock import Clock, utc_now
from app.config import Settings

log = logging.getLogger("app")

_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


def create_app(settings: Settings | None = None, *, clock: Clock = utc_now) -> FastAPI:
    """Build the app. Settings default to the environment, so bad config fails at startup."""
    settings = Settings.from_env() if settings is None else settings
    _configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("starting with %s", settings)
        # #2: create the schema at settings.db_path
        # #8: clear stale claims before workers start
        # #7: start settings.worker_count worker tasks when worker_count > 0
        yield
        # #7: cancel worker tasks

    app = FastAPI(title="Ticket classifier", lifespan=lifespan)
    app.state.settings = settings
    app.state.clock = clock
    install_error_handlers(app)
    app.include_router(router)
    return app


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"), headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def on_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's "input" and "ctx" are dropped so request content is never echoed back.
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return error_response(422, "validation_error", "Request validation failed.", details)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def on_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Starlette re-raises after sending this response, so the server logs the traceback.
        return error_response(500, "internal_error", "Internal server error.")


def _configure_logging() -> None:
    # Only when nothing else (uvicorn's --log-config, pytest) has configured the root logger.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
