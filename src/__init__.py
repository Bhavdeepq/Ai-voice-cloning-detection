"""Inference API for the SIH V4 voice-cloning detector."""

__all__ = ["VoiceDetector"]


def __getattr__(name: str):
    if name == "VoiceDetector":
        from .detector import VoiceDetector

        return VoiceDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
