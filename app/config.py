"""Settings read from environment variables and validated once at startup."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

_FLOAT_FIELDS = (
    "llm_timeout_s",
    "lease_timeout_s",
    "poll_interval_s",
    "backoff_base_s",
    "fake_llm_failure_rate",
)
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """Raised at startup when an environment variable is malformed or out of bounds."""


@dataclass(frozen=True, slots=True)
class Settings:
    db_path: str = "tickets.db"
    worker_count: int = 4
    max_attempts: int = 3
    llm_timeout_s: float = 10.0
    lease_timeout_s: float = 60.0
    poll_interval_s: float = 0.5
    backoff_base_s: float = 1.0
    fake_llm_seed: int = 42
    fake_llm_failure_rate: float = 0.2
    fake_llm_gullible: bool = True

    def __post_init__(self) -> None:
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ConfigError(f"{name.upper()} must be a finite number, got {value!r}")
        if not self.db_path.strip():
            raise ConfigError("DB_PATH must not be empty")
        if self.worker_count < 0:
            raise ConfigError(f"WORKER_COUNT must be 0 or greater, got {self.worker_count}")
        if self.max_attempts < 1:
            raise ConfigError(f"MAX_ATTEMPTS must be at least 1, got {self.max_attempts}")
        if self.llm_timeout_s <= 0:
            raise ConfigError(f"LLM_TIMEOUT_S must be greater than 0, got {self.llm_timeout_s}")
        if self.lease_timeout_s <= self.llm_timeout_s:
            raise ConfigError(
                f"LEASE_TIMEOUT_S ({self.lease_timeout_s}) must be greater than "
                f"LLM_TIMEOUT_S ({self.llm_timeout_s})"
            )
        if self.poll_interval_s <= 0:
            raise ConfigError(f"POLL_INTERVAL_S must be greater than 0, got {self.poll_interval_s}")
        if self.backoff_base_s < 0:
            raise ConfigError(f"BACKOFF_BASE_S must be 0 or greater, got {self.backoff_base_s}")
        if not 0.0 <= self.fake_llm_failure_rate <= 1.0:
            raise ConfigError(
                f"FAKE_LLM_FAILURE_RATE must be between 0 and 1, got {self.fake_llm_failure_rate}"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` (default ``os.environ``); unset or blank values use the defaults."""
        env = os.environ if env is None else env
        # slots=True turns class attributes into slot descriptors, so read defaults from an instance.
        d = cls()
        return cls(
            db_path=_str(env, "DB_PATH", d.db_path),
            worker_count=_int(env, "WORKER_COUNT", d.worker_count),
            max_attempts=_int(env, "MAX_ATTEMPTS", d.max_attempts),
            llm_timeout_s=_float(env, "LLM_TIMEOUT_S", d.llm_timeout_s),
            lease_timeout_s=_float(env, "LEASE_TIMEOUT_S", d.lease_timeout_s),
            poll_interval_s=_float(env, "POLL_INTERVAL_S", d.poll_interval_s),
            backoff_base_s=_float(env, "BACKOFF_BASE_S", d.backoff_base_s),
            fake_llm_seed=_int(env, "FAKE_LLM_SEED", d.fake_llm_seed),
            fake_llm_failure_rate=_float(env, "FAKE_LLM_FAILURE_RATE", d.fake_llm_failure_rate),
            fake_llm_gullible=_bool(env, "FAKE_LLM_GULLIBLE", d.fake_llm_gullible),
        )


def _raw(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = _raw(env, name)
    return default if raw is None else raw


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _raw(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _raw(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number, got {raw!r}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _raw(env, name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(f"{name} must be one of 1/0, true/false, yes/no, on/off; got {raw!r}")
