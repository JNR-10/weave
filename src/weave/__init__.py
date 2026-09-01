"""Weave — runtime-adaptive placement for distributed LLM inference on Apple Silicon.

Weave sits above MLX Distributed. It does not execute the model; it decides how execution
should be configured — which Macs participate in a workload, and how many transformer layers
each one runs — and revises that decision at safe request boundaries as conditions change.
"""

__version__ = "0.0.0"
