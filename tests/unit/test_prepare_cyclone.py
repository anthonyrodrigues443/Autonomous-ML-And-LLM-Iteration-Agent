"""The storm example's download step, with no network: what it retries, what it fetches
again, and what a failed write leaves behind."""

from __future__ import annotations

import http.client
import importlib.util
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

_PREPARE = Path(__file__).resolve().parents[2] / "examples" / "cyclone_wind" / "prepare.py"
_ROW = {"file": "images/00000.jpg", "Image ID": "abc_000"}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cyclone_prepare", _PREPARE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("L", (64, 64), 120).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = _load()
    monkeypatch.setattr(module, "HERE", tmp_path)
    monkeypatch.setattr(module, "IMAGES", tmp_path / "images")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    return module


class _Response:
    def __init__(self, body: bytes, *, cut: bool = False) -> None:
        self.headers = {"Content-Length": str(len(body))}
        self._body = body
        self._cut = cut

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        if self._cut:
            raise http.client.IncompleteRead(self._body[:10], len(self._body) - 10)
        return self._body


def test_a_response_cut_short_is_retried(
    prepare: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _jpeg()
    calls: list[str] = []

    def urlopen(request: Any, timeout: float) -> _Response:
        calls.append(request.full_url)
        return _Response(body, cut=len(calls) == 1)

    monkeypatch.setattr(prepare.urllib.request, "urlopen", urlopen)
    assert prepare._get("https://example.invalid/a.jpg") == body
    assert len(calls) == 2


def test_a_truncated_frame_already_on_disk_is_fetched_again(
    prepare: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _jpeg()
    fetched: list[str] = []
    monkeypatch.setattr(prepare, "_get", lambda url, **_: fetched.append(url) or body)
    frame = prepare.HERE / _ROW["file"]
    frame.parent.mkdir()
    frame.write_bytes(body[: len(body) // 2])

    prepare.download([_ROW])

    assert frame.read_bytes() == body
    assert len(fetched) == 1


def test_a_whole_frame_already_on_disk_is_not_fetched_again(
    prepare: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[str] = []
    monkeypatch.setattr(prepare, "_get", lambda url, **_: fetched.append(url) or b"")
    frame = prepare.HERE / _ROW["file"]
    frame.parent.mkdir()
    frame.write_bytes(_jpeg())

    prepare.download([_ROW])

    assert fetched == []


def test_a_write_that_fails_part_way_leaves_no_frame_under_its_name(
    prepare: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _jpeg()
    monkeypatch.setattr(prepare, "_get", lambda _url, **_: body)
    real = Path.write_bytes

    def disk_full(self: Path, data: bytes) -> int:
        real(self, data[: len(data) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", disk_full)
    with pytest.raises(OSError, match="No space left"):
        prepare.download([_ROW])

    assert not (prepare.HERE / _ROW["file"]).exists()


def test_a_manifest_that_does_not_match_says_what_to_delete(
    prepare: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{**_ROW, "Storm ID": "abc", "label": "30", "side": "train"}]
    monkeypatch.setattr(prepare, "label_rows", list)
    monkeypatch.setattr(prepare, "choose", lambda _rows, _every: rows)
    monkeypatch.setattr(prepare, "download", lambda _kept: "0" * 64)
    monkeypatch.setattr("sys.argv", ["prepare.py"])
    with pytest.raises(SystemExit, match=r"delete .*images and rerun"):
        prepare.main()
