"""Running a chosen placement on real Macs: launch order, split-aware loading, generation.

Spec 0005. The portable half (weave.runtime.launch) turns a logical plan into what one MLX
launch needs and checks what each rank reports back; it imports on Intel and in CI. The MLX
half (weave.runtime.mlx_pipeline) imports MLX inside its functions and runs only on Apple
Silicon. Until a two-Mac run passes gates P2-B to P2-D, everything here is unverified on real
hardware.
"""
