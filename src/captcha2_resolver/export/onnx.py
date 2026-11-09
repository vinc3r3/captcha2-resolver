"""Convert the exported PaddleOCR rec model to ONNX (+ dict.txt), and check parity with Paddle.

Input : a Paddle export dir (inference.json|pdmodel + inference.pdiparams + inference.yml) or its .zip
Output: model.onnx  - input "x": float32 [N, 3, 48, W], output: softmax probs [N, T, num_classes]
        dict.txt    - CTC alphabet; class 0 is blank, class i is dict line i-1
        inference.yml (copied; also embeds the dict)

Gotcha this works around: Paddle >= 3.3 exports nn.Linear as the fused `pd_op.linear_v2`, which
paddle2onnx 2.1 cannot convert. Each one is rewritten as matmul + add before conversion.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _unpack(src: Path, tmp: Path) -> Path:
    if src.is_dir():
        return src
    with zipfile.ZipFile(src) as zf:
        zf.extractall(tmp)
    return tmp


def _model_file(model_dir: Path) -> str:
    for name in ("inference.json", "inference.pdmodel"):  # Paddle 3.x PIR / legacy
        if (model_dir / name).exists():
            return name
    raise FileNotFoundError(f"no inference.json / inference.pdmodel in {model_dir}")


def load_dict(model_dir: Path) -> list[str]:
    import yaml

    post = yaml.safe_load((model_dir / "inference.yml").read_text(encoding="utf-8"))["PostProcess"]
    assert post["name"] == "CTCLabelDecode", post["name"]
    return [str(c) for c in post["character_dict"]]


def replace_linear_v2(model_dir: Path, out_dir: Path) -> bool:
    import paddle

    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    prog, _, _ = paddle.static.load_inference_model(str(model_dir / "inference"), exe)
    block = prog.global_block()
    ops = [op for op in block.ops if op.name() == "pd_op.linear_v2"]
    if not ops:
        return False
    with paddle.static.program_guard(prog):
        for op in ops:
            x, w, b = op.operands_source()
            paddle.pir.set_insertion_point(op)
            out = paddle._C_ops.add(paddle._C_ops.matmul(x, w, False, op.attrs()["transpose_weight"]), b)
            op.result(0).replace_all_uses_with(out)
            block.remove_op(op)
    feeds = [op.result(0) for op in block.ops if op.name() == "pd_op.data"]
    fetches = [op.operand_source(0) for op in block.ops if op.name() == "pd_op.fetch"]
    paddle.static.save_inference_model(str(out_dir / "inference"), feeds, fetches, exe, program=prog)
    print(f"replaced {len(ops)} linear_v2 ops with matmul + add")
    return True


def paddle_probs(model_dir: Path, x: np.ndarray) -> np.ndarray:
    import paddle.inference as pi

    model_file = _model_file(model_dir)
    config = pi.Config(str(model_dir / model_file), str(model_dir / "inference.pdiparams"))
    config.disable_gpu()
    pred = pi.create_predictor(config)
    pred.get_input_handle(pred.get_input_names()[0]).copy_from_cpu(x)
    pred.run()
    return pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu()


def convert(src: Path, out: Path, opset: int = 14, check_image: Path | None = None) -> Path:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        model_dir = _unpack(src, tmp / "src")
        model_file = _model_file(model_dir)
        chars = load_dict(model_dir)
        convert_dir = model_dir
        if model_file == "inference.json":
            (tmp / "fixed").mkdir()
            if replace_linear_v2(model_dir, tmp / "fixed"):
                convert_dir = tmp / "fixed"
        out.mkdir(parents=True, exist_ok=True)
        onnx_path = out / "model.onnx"
        cmd = ["paddle2onnx", "--model_dir", str(convert_dir), "--model_filename", model_file,
               "--params_filename", "inference.pdiparams", "--save_file", str(onnx_path),
               "--opset_version", str(opset), "--enable_onnx_checker", "True"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        (out / "dict.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
        shutil.copy2(model_dir / "inference.yml", out / "inference.yml")
        print(f"saved {onnx_path} + dict.txt ({len(chars)} chars)")

        if check_image is not None:
            import cv2

            from captcha2_resolver.infer import CaptchaSolver, ctc_decode, preprocess

            solver = CaptchaSolver(out)
            x = preprocess(cv2.imread(str(check_image)))
            onnx_p = solver.probs(x)
            print("ONNX  :", *ctc_decode(onnx_p[0], solver.chars))
            try:
                pp = paddle_probs(model_dir, x)
                print("Paddle:", *ctc_decode(pp[0], chars))
                print("max |onnx - paddle| =", float(np.abs(onnx_p - pp).max()))
            except ImportError:
                print("paddle not installed: skipped parity check")
    return onnx_path
