"""Regression tests for the labeling, Hub publishing, labels-file and zip-export fixes."""

import json
import zipfile
from pathlib import Path

import pytest

from captcha2_resolver import cli
from captcha2_resolver.data import hub, merge
from captcha2_resolver.export.onnx import _unpack
from captcha2_resolver.label import rucaptcha as rc


@pytest.fixture
def images(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for n in ("a.jpg", "b.jpg"):
        (d / n).write_bytes(b"jpg")
    return d


def _label(images, runs, *extra):
    cli.main(["label", "rucaptcha", "--images", str(images), "--runs", str(runs), *extra])


# ---------------------------------------------------------------- labeling
def test_missing_api_key_does_not_mark_images_as_sent(images, tmp_path, monkeypatch):
    monkeypatch.delenv("RUCAPTCHA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit):
        _label(images, runs)
    assert rc.already_sent(runs) == set()


def test_dry_run_with_resume_sends_nothing(images, tmp_path, monkeypatch, capsys):
    run = rc.create_run(["a.jpg", "b.jpg"], tmp_path / "runs")
    monkeypatch.setenv("RUCAPTCHA_API_KEY", "k")
    monkeypatch.setattr(rc, "solve_run", lambda *a, **k: pytest.fail("dry run sent requests"))
    _label(images, tmp_path / "runs", "--resume", str(run), "--dry-run")
    assert capsys.readouterr().out.split() == ["a.jpg", "b.jpg"]


class FakeClient:
    def __init__(self, poll_exc):
        self.poll_exc, self.created = poll_exc, 0

    def create_task(self, image):
        self.created += 1
        return 100 + self.created

    def get_result(self, task_id):
        raise self.poll_exc


def test_permanent_poll_error_is_resubmitted_on_resume(images, tmp_path):
    run = rc.create_run(["a.jpg"], tmp_path / "runs")
    client = FakeClient(RuntimeError("ERROR_CAPTCHA_UNSOLVABLE"))
    (r,) = rc.solve_run(run, images, client)
    assert r["status"] == "error" and r["task_id"] is None and r["failed_task_ids"] == [101]
    rc.solve_run(run, images, client)
    assert client.created == 2


def test_timeout_keeps_task_id_so_it_is_not_paid_twice(images, tmp_path):
    run = rc.create_run(["a.jpg"], tmp_path / "runs")
    client = FakeClient(TimeoutError("slow"))
    (r,) = rc.solve_run(run, images, client)
    assert r["status"] == "error" and r["task_id"] == 101
    rc.solve_run(run, images, client)
    assert client.created == 1


# ---------------------------------------------------------------- hub
class FakeApi:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, k))


@pytest.mark.parametrize("gated", ["manual", "private"])
def test_publish_makes_repo_private_before_upload(tmp_path, gated):
    api = FakeApi()
    hub._publish(api, "u/r", "dataset", tmp_path, gated, "msg")
    names = [c[0] for c in api.calls]
    assert names.index("update_repo_settings") < names.index("upload_folder")
    assert api.calls[names.index("update_repo_settings")][1]["private"] is True
    made_public = any(k.get("private") is False for _, k in api.calls)
    assert made_public == (gated == "manual")


def test_publish_rejects_bad_gated_before_touching_the_hub(tmp_path):
    api = FakeApi()
    with pytest.raises(ValueError):
        hub._publish(api, "u/r", "model", tmp_path, "public", "msg")
    assert api.calls == []


def test_push_model_requires_model_files_and_metrics(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.onnx"):
        hub.push_model("u/r", tmp_path, tmp_path / "m.json", dry_run=True)
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "dict.txt").write_text("a\n")
    with pytest.raises(FileNotFoundError, match="c2r evaluate"):
        hub.push_model("u/r", tmp_path, tmp_path / "m.json", dry_run=True)


# ---------------------------------------------------------------- labels / export
def test_load_labels_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="c2r data merge"):
        merge.load_labels(tmp_path / "labels.csv")


@pytest.mark.parametrize("prefix", ["", "paddle/"])
def test_unpack_finds_model_at_root_or_in_a_folder(tmp_path, prefix):
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(f"__MACOSX/{prefix}inference.json", "")
        zf.writestr(f"{prefix}inference.json", json.dumps({}))
        zf.writestr(f"{prefix}inference.pdiparams", "")
    out = tmp_path / "out"
    assert _unpack(z, out) == (out / prefix.rstrip("/") if prefix else out)
    assert (Path(_unpack(z, tmp_path / "o2")) / "inference.pdiparams").exists()
