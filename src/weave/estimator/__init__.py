"""Workload and KV-Cache Memory Estimator — what the next request will cost.

Predicts per-device memory demand ahead of execution from the request's context length,
requested output length, and the current concurrency level, combined with the model's
architecture (layer count, attention scheme, dtype).

The estimate is what allows placement to be chosen *before* a request runs, instead of
discovering a device is short on memory once it has already failed.
"""
