"""specdraft — speculative-draft training framework for the Qwen family (DFLASH/DSPARK/DFLASH2).

Wraps vllm-speculators as the training engine and hardens the full pipeline
(regen → pretokenize → prepare → hsextract → train → convert → serve → bench)
with Qwen presets.
"""
__version__ = "0.1.1"
