"""Paid labeling with RuCaptcha (human workers, `ImageToTextTask`).

Design, kept from the original tool because it saved money and data:
  * A *run* is a folder `runs/rucaptcha/<UTC timestamp>_<id>/` with `scope.json` (the images
    chosen for this run) and `results.json` (one record per image: task id, status, text).
  * Scope selection never re-sends an image that is already in any previous run.
  * All tasks are submitted first, then polled; `results.json` is checkpointed (atomic write)
    after every change, so an interrupted run can be resumed with `--resume` without paying twice.
  * Records store file *names* (relative to the image dir), never absolute paths, and never the
    image bytes or the API key. Note: RuCaptcha responses include the worker IP, so run folders
    stay local (gitignored) and only the consolidated labels are published.

Task settings that mattered: fixed `minLength=maxLength=6`, a short Russian instruction
("введите текст, который вы видите на изображении" = "type the text you see in the image")
and `languagePool="rn"`. Even so ~7% of answers came back in Cyrillic (workers typing
look-alike letters on a Russian keyboard); those are dropped during consolidation.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

CREATE_URL = "https://api.rucaptcha.com/createTask"
RESULT_URL = "https://api.rucaptcha.com/getTaskResult"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
INSTRUCTION = "введите текст, который вы видите на изображении"


class RucaptchaClient:
    def __init__(self, api_key: str, length: int = 6):
        if not api_key:
            raise ValueError("Set RUCAPTCHA_API_KEY in the environment or .env")
        self.api_key, self.length = api_key, length

    def _post(self, url: str, body: dict) -> dict:
        import requests

        resp = requests.post(url, json=body, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if not isinstance(result, dict) or result.get("errorId") != 0:
            raise RuntimeError(f"RuCaptcha error: {result}")
        return result

    def create_task(self, image: Path) -> int:
        body = {
            "clientKey": self.api_key,
            "task": {
                "type": "ImageToTextTask",
                "body": base64.b64encode(image.read_bytes()).decode("ascii"),
                "minLength": self.length,
                "maxLength": self.length,
                "comment": INSTRUCTION,
            },
            "languagePool": "rn",
        }
        task_id = self._post(CREATE_URL, body).get("taskId")
        if not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError("RuCaptcha returned no valid taskId")
        return task_id

    def get_result(self, task_id: int) -> dict:
        return self._post(RESULT_URL, {"clientKey": self.api_key, "taskId": task_id})


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def already_sent(runs_dir: Path) -> set[str]:
    sent: set[str] = set()
    for scope in runs_dir.glob("*/scope.json"):
        sent.update(Path(p).name for p in json.loads(scope.read_text(encoding="utf-8")))
    return sent


def select_scope(images_dir: Path, runs_dir: Path, size: int) -> list[str]:
    """Next `size` image names not included in any saved run. Sends no requests."""
    sent = already_sent(runs_dir)
    names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [n for n in names if n not in sent][:size]


def create_run(names: list[str], runs_dir: Path) -> Path:
    if not names:
        raise ValueError("empty scope")
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "scope.json", names)
    _write_json(run_dir / "results.json",
                [{"filename": n, "task_id": None, "status": "pending", "text": None} for n in names])
    return run_dir


def pending(run_dir: Path) -> list[str]:
    """Names in a run that `solve_run` would still submit or poll (i.e. not ready)."""
    records = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    return [r["filename"] for r in records if r["status"] != "ready"]


def solve_run(run_dir: Path, images_dir: Path, client: RucaptchaClient,
              timeout: float = 180, poll: float = 3) -> list[dict]:
    """Submit every pending record, then poll until ready/error. Resumable."""
    results_path = run_dir / "results.json"
    records = json.loads(results_path.read_text(encoding="utf-8"))

    def checkpoint():
        _write_json(results_path, records)

    for r in records:  # 1) submit
        if r["status"] == "ready" or r.get("task_id"):
            continue
        try:
            r.update(task_id=client.create_task(images_dir / r["filename"]), status="submitted")
            print(f'{r["filename"]}: task {r["task_id"]}', flush=True)
        except Exception as exc:  # keep going; failed records can be retried with --resume
            r.update(status="error", error=str(exc))
            print(f'{r["filename"]}: error {exc}', flush=True)
        checkpoint()

    for r in records:  # 2) poll
        if r["status"] == "ready" or not r.get("task_id"):
            continue
        deadline = time.monotonic() + timeout
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'task {r["task_id"]} timed out')
                res = client.get_result(r["task_id"])
                if res.get("status") == "ready":
                    text = (res.get("solution") or {}).get("text")
                    if not text:
                        raise RuntimeError("ready task returned no text")
                    r.update(status="ready", text=text, cost=res.get("cost"))
                    r.pop("error", None)
                    print(f'{r["filename"]}: {text}', flush=True)
                    break
                time.sleep(poll)
        except RuntimeError as exc:
            # The server rejected the task for good (unsolvable, expired id, empty answer). Drop the
            # task id so --resume submits the image again; keep the old id for the cost audit.
            r.setdefault("failed_task_ids", []).append(r["task_id"])
            r.update(status="error", error=str(exc), task_id=None)
            print(f'{r["filename"]}: error {exc} (will be resubmitted on --resume)', flush=True)
        except Exception as exc:
            # Timeout or network error: the task may still finish, so keep polling it on --resume
            # instead of paying for it twice.
            r.update(status="error", error=str(exc))
            print(f'{r["filename"]}: error {exc}', flush=True)
        checkpoint()
    return records
