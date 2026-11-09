"""captcha2-resolver: collect -> label -> correct -> train (PaddleOCR) -> deploy (ONNX)."""

__version__ = "0.1.0"
__all__ = ["CaptchaSolver"]


def __getattr__(name):  # lazy, so `c2r collect` etc. don't import onnxruntime
    if name == "CaptchaSolver":
        from captcha2_resolver.infer import CaptchaSolver

        return CaptchaSolver
    raise AttributeError(name)
