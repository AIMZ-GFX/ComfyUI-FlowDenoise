from .nodes import TemporalFlowAverage, ExtractNoise, SelectiveDenoise

NODE_CLASS_MAPPINGS = {
    "TemporalFlowAverage": TemporalFlowAverage,
    "ExtractNoise": ExtractNoise,
    "SelectiveDenoise": SelectiveDenoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TemporalFlowAverage": "Temporal Flow Average",
    "ExtractNoise": "Extract Noise (Chroma/Luma)",
    "SelectiveDenoise": "Selective Denoise",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
