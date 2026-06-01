"""Live integration: the E2BCodeRunner against a real e2b sandbox.

Skips unless E2B_API_KEY is set and the `[sandbox]` extra is installed. Verifies
the round-trip (upload input, run script, read output, teardown) on real e2b.
This is the only thing that can confirm the documented API calls are right; the
unit tests use a fake sandbox.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from iterate.adapters.compute.runner import E2BCodeRunner

_HAS_E2B = importlib.util.find_spec("e2b_code_interpreter") is not None
_HAS_KEY = bool(os.environ.get("E2B_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not (_HAS_E2B and _HAS_KEY), reason="needs [sandbox] extra + E2B_API_KEY")
def test_e2b_runner_round_trip_live() -> None:
    script = (
        "data = open('in.txt').read()\n"
        "open('out.txt', 'w').write(data.upper())\n"
        "print('ok')\n"
    )
    result = E2BCodeRunner().run(
        script,
        inputs={"in.txt": b"hello"},
        outputs=["out.txt"],
        timeout=60,
    )
    assert result.succeeded
    assert b"HELLO" in result.outputs["out.txt"]
