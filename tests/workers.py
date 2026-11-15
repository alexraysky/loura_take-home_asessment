"""Helpers for the lifespan tests that run real workers (test_app.py, test_reclassify.py)."""

import asyncio
import time

from fastapi.testclient import TestClient

DEADLINE_S = 5.0


class GateLLM:
    """Parks every call forever; counts the parked calls."""

    def __init__(self) -> None:
        self.parked = 0

    async def complete(self, system: str, user: str) -> str:
        self.parked += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def wait_until(predicate, what: str):
    deadline = time.monotonic() + DEADLINE_S
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


def wait_for_status(client: TestClient, id: str, status: str) -> dict:
    def check():
        body = client.get(f"/tickets/{id}").json()
        return body if body["status"] == status else None

    return wait_until(check, f"{id} to be {status}")
