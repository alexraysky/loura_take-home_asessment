"""FastAPI app factory, lifespan and error handlers."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import ApiError, ErrorBody, ErrorEnvelope, router
from app.clock import Clock, utc_now
from app.config import Settings

log = logging.getLogger("app")

_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}
# Statuses that must not carry a body.
_NO_BODY = {204, 304}


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
    async def on_api_error(_: Request, exc: ApiError) -> Response:
        if exc.status_code in _NO_BODY:
            return Response(status_code=exc.status_code)
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's "input" and "ctx" are dropped so request content is never echoed back.
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return error_response(422, "validation_error", "Request validation failed.", details)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(_: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code in _NO_BODY:
            return Response(status_code=exc.status_code, headers=exc.headers)
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        if isinstance(exc.detail, str) and exc.detail:
            return error_response(exc.status_code, code, exc.detail, headers=exc.headers)
        # Structured detail goes to "details"; the message stays a readable string.
        details = None if isinstance(exc.detail, str) else exc.detail
        try:
            message = HTTPStatus(exc.status_code).phrase
        except ValueError:
            message = "HTTP error"
        return error_response(exc.status_code, code, message, details, headers=exc.headers)

    @app.exception_handler(Exception)
    async def on_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Starlette re-raises after sending this response, so the server logs the traceback.
        return error_response(500, "internal_error", "Internal server error.")


def _configure_logging() -> None:
    # uvicorn configures only its own loggers; give the "app" logger a handler so its INFO lines show.
    # Skipped when the root logger is already set up (pytest, --log-config) and left alone otherwise.
    if logging.getLogger().handlers or log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
