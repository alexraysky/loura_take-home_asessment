"""Settings read from environment variables and validated once at startup."""

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

_INT_FIELDS = ("worker_count", "max_attempts", "fake_llm_seed")
_FLOAT_FIELDS = (
    "llm_timeout_s",
    "lease_timeout_s",
    "poll_interval_s",
    "backoff_base_s",
    "fake_llm_failure_rate",
)
# Upper bounds keep later arithmetic (lease expiry, backoff, task count) well inside sane ranges.
MAX_WORKER_COUNT = 100
MAX_ATTEMPTS = 20
MAX_SECONDS = 86_400.0
MIN_POLL_INTERVAL_S = 0.001
MAX_SEED = 10**18 - 1
_INT_RE = re.compile(r"[+-]?[0-9]{1,18}")
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
        self._check_types()
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ConfigError(f"{name.upper()} must be a finite number, got {value!r}")
        if not self.db_path.strip():
            raise ConfigError("DB_PATH must not be empty")
        if not 0 <= self.worker_count <= MAX_WORKER_COUNT:
            raise ConfigError(
                f"WORKER_COUNT must be between 0 and {MAX_WORKER_COUNT}, got {self.worker_count}"
            )
        if not 1 <= self.max_attempts <= MAX_ATTEMPTS:
            raise ConfigError(f"MAX_ATTEMPTS must be between 1 and {MAX_ATTEMPTS}, got {self.max_attempts}")
        for name in ("llm_timeout_s", "lease_timeout_s", "poll_interval_s", "backoff_base_s"):
            if getattr(self, name) > MAX_SECONDS:
                raise ConfigError(f"{name.upper()} must be at most {MAX_SECONDS:g}, got {getattr(self, name)}")
        if self.llm_timeout_s <= 0:
            raise ConfigError(f"LLM_TIMEOUT_S must be greater than 0, got {self.llm_timeout_s}")
        if self.lease_timeout_s <= self.llm_timeout_s:
            raise ConfigError(
                f"LEASE_TIMEOUT_S ({self.lease_timeout_s}) must be greater than "
                f"LLM_TIMEOUT_S ({self.llm_timeout_s})"
            )
        if self.poll_interval_s < MIN_POLL_INTERVAL_S:
            raise ConfigError(
                f"POLL_INTERVAL_S must be at least {MIN_POLL_INTERVAL_S:g}, got {self.poll_interval_s}"
            )
        if abs(self.fake_llm_seed) > MAX_SEED:
            raise ConfigError(f"FAKE_LLM_SEED must have at most 18 digits, got {self.fake_llm_seed}")
        if self.backoff_base_s < 0:
            raise ConfigError(f"BACKOFF_BASE_S must be 0 or greater, got {self.backoff_base_s}")
        if not 0.0 <= self.fake_llm_failure_rate <= 1.0:
            raise ConfigError(
                f"FAKE_LLM_FAILURE_RATE must be between 0 and 1, got {self.fake_llm_failure_rate}"
            )

    def _check_types(self) -> None:
        # Direct construction (tests, later issues) gets the same guarantees as from_env.
        if not isinstance(self.db_path, str):
            raise ConfigError(f"DB_PATH must be a string, got {self.db_path!r}")
        if not isinstance(self.fake_llm_gullible, bool):
            raise ConfigError(f"FAKE_LLM_GULLIBLE must be a boolean, got {self.fake_llm_gullible!r}")
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{name.upper()} must be an integer, got {value!r}")
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"{name.upper()} must be a number, got {value!r}")
            try:
                object.__setattr__(self, name, float(value))
            except OverflowError:
                raise ConfigError(f"{name.upper()} must be a finite number, got {value!r}") from None

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
    # int() alone would also accept "1_000", non-ASCII digits and numbers too long to convert.
    if not _INT_RE.fullmatch(raw):
        raise ConfigError(f"{name} must be an integer, got {raw!r}")
    return int(raw)


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _raw(env, name)
    if raw is None:
        return default
    try:
        # float() alone would also accept "1_0" and non-ASCII digits.
        if "_" in raw or not raw.isascii():
            raise ValueError(raw)
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
