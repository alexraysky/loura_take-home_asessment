import dataclasses

import pytest

from app.config import ConfigError, Settings
from app.main import create_app

ALL_VARS = {
    "DB_PATH": "/tmp/other.db",
    "WORKER_COUNT": "2",
    "MAX_ATTEMPTS": "5",
    "LLM_TIMEOUT_S": "2.5",
    "LEASE_TIMEOUT_S": "30",
    "POLL_INTERVAL_S": "0.1",
    "BACKOFF_BASE_S": "0",
    "FAKE_LLM_SEED": "7",
    "FAKE_LLM_FAILURE_RATE": "0",
    "FAKE_LLM_GULLIBLE": "false",
}


def test_defaults_match_documented_table():
    assert Settings.from_env({}) == Settings(
        db_path="tickets.db",
        worker_count=4,
        max_attempts=3,
        llm_timeout_s=10.0,
        lease_timeout_s=60.0,
        poll_interval_s=0.5,
        backoff_base_s=1.0,
        fake_llm_seed=42,
        fake_llm_failure_rate=0.2,
        fake_llm_gullible=True,
    )


def test_every_variable_is_read_and_converted():
    s = Settings.from_env(ALL_VARS)
    assert s == Settings(
        db_path="/tmp/other.db",
        worker_count=2,
        max_attempts=5,
        llm_timeout_s=2.5,
        lease_timeout_s=30.0,
        poll_interval_s=0.1,
        backoff_base_s=0.0,
        fake_llm_seed=7,
        fake_llm_failure_rate=0.0,
        fake_llm_gullible=False,
    )
    assert type(s.worker_count) is int
    assert type(s.lease_timeout_s) is float


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_values_fall_back_to_defaults(blank):
    assert Settings.from_env({name: blank for name in ALL_VARS}) == Settings()


def test_unrelated_variables_are_ignored():
    assert Settings.from_env({"PATH": "/bin", "WORKERS": "9"}) == Settings()


def test_surrounding_whitespace_is_stripped():
    s = Settings.from_env({"WORKER_COUNT": " 3 ", "DB_PATH": " x.db ", "FAKE_LLM_GULLIBLE": " No "})
    assert (s.worker_count, s.db_path, s.fake_llm_gullible) == (3, "x.db", False)


def test_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("WORKER_COUNT", "0")
    assert Settings.from_env().worker_count == 0


def test_worker_count_zero_is_accepted():
    assert Settings.from_env({"WORKER_COUNT": "0"}).worker_count == 0


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "On"])
def test_truthy_booleans(raw):
    assert Settings.from_env({"FAKE_LLM_GULLIBLE": raw}).fake_llm_gullible is True


@pytest.mark.parametrize("raw", ["0", "false", "False", "no", "OFF"])
def test_falsy_booleans(raw):
    assert Settings.from_env({"FAKE_LLM_GULLIBLE": raw}).fake_llm_gullible is False


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"LEASE_TIMEOUT_S": "10"}, "LEASE_TIMEOUT_S (10.0) must be greater than LLM_TIMEOUT_S (10.0)"),
        ({"LEASE_TIMEOUT_S": "5"}, "LEASE_TIMEOUT_S (5.0) must be greater than LLM_TIMEOUT_S (10.0)"),
        ({"LLM_TIMEOUT_S": "90"}, "LEASE_TIMEOUT_S (60.0) must be greater than LLM_TIMEOUT_S (90.0)"),
    ],
)
def test_lease_must_exceed_llm_timeout(env, fragment):
    with pytest.raises(ConfigError) as info:
        Settings.from_env(env)
    assert fragment in str(info.value)


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("WORKER_COUNT", "-1"),
        ("MAX_ATTEMPTS", "0"),
        ("LLM_TIMEOUT_S", "0"),
        ("LLM_TIMEOUT_S", "-1"),
        ("POLL_INTERVAL_S", "0"),
        ("BACKOFF_BASE_S", "-0.5"),
        ("FAKE_LLM_FAILURE_RATE", "1.5"),
        ("FAKE_LLM_FAILURE_RATE", "-0.1"),
    ],
)
def test_out_of_bounds_values_name_the_variable(name, raw):
    with pytest.raises(ConfigError, match=name):
        Settings.from_env({name: raw})


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("WORKER_COUNT", "four"),
        ("WORKER_COUNT", "2.5"),
        ("FAKE_LLM_SEED", "0x10"),
        ("LLM_TIMEOUT_S", "ten"),
        ("FAKE_LLM_GULLIBLE", "maybe"),
    ],
)
def test_malformed_values_name_the_variable(name, raw):
    with pytest.raises(ConfigError, match=name) as info:
        Settings.from_env({name: raw})
    assert repr(raw) in str(info.value)


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity"])
@pytest.mark.parametrize("name", ["LLM_TIMEOUT_S", "LEASE_TIMEOUT_S", "POLL_INTERVAL_S", "BACKOFF_BASE_S", "FAKE_LLM_FAILURE_RATE"])
def test_non_finite_floats_are_rejected(name, raw):
    with pytest.raises(ConfigError, match=f"{name} must be a finite number"):
        Settings.from_env({name: raw})


def test_direct_construction_is_validated_too():
    with pytest.raises(ConfigError, match="LEASE_TIMEOUT_S"):
        Settings(lease_timeout_s=float("nan"))
    with pytest.raises(ConfigError, match="DB_PATH"):
        Settings(db_path="  ")


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


def test_settings_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().worker_count = 1  # type: ignore[misc]


def test_create_app_checks_environment_at_startup(monkeypatch):
    monkeypatch.setenv("LEASE_TIMEOUT_S", "5")
    monkeypatch.delenv("LLM_TIMEOUT_S", raising=False)
    with pytest.raises(ConfigError, match="LEASE_TIMEOUT_S"):
        create_app()
