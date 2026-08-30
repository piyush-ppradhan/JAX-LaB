# Experimental implementations

This package contains performance experiments that are not part of JAX-LaB's
stable API. Experimental kernels must reject unsupported configurations instead
of silently changing simulation behavior.

Current scope:

- `pallas_kernels.py`: D2Q9, D3Q19, and D3Q27 Pallas BGK and MRT collision
  kernels and optional fused collision/streaming building blocks.
- `pallas_sim.py`: `PallasBGK` and `PallasMRT`, `LBMBase`-compatible
  multi-GPU classes that shard the X dimension and exchange only crossing
  populations, preserving initialization, `run`, I/O, checkpointing, and
  output semantics while using direction-major population storage
  internally. Both share their non-collision-specific machinery (SoA
  streaming, boundary dispatch, `step`, I/O) through `_PallasSoASimMixin`;
  each supplies its own `_build_local_pallas_collision`.
- `PallasMRT` drives the fused Pallas collision from `MRTSim`'s own
  `collision_terms` (the sparse per-output-direction decomposition of
  `M @ S @ M_inv`), matching core `MRTSim.collision` exactly (including its
  `s_v` quirk - `MRTSim.__init__` always forces `omega = 1.0` and derives
  `s_v = self.omega`, so `s_v` cannot be set independently of `1.0` through
  the public kwargs; only `omega = 1.0` (with a matching uniform `S`
  diagonal) degenerates exactly to a corresponding BGK simulation). With an
  identity `M` and a uniform `S` diagonal, it reproduces `PallasBGK`
  exactly - verified bit-for-bit at `omega=1.0` on a real 2-GPU cavity run,
  and cross-checked against core `MRTSim.collision` directly for both an
  identity and the real (off-diagonal) D3Q19 MRT moment matrix.
- Both simulation classes always use Pallas collision followed by the normal
  collision/BC/streaming/BC ordering. `BounceBack`, `BounceBackHalfway`,
  `BounceBackMoving`, `InterpolatedBounceBackBouzidi`,
  `InterpolatedBounceBackDifferentiable`, `DoNothing`, `EquilibriumBC`, `ZouHe`,
  `Regularized`, `ConvectiveOutflow`, `NonEquilibriumExtrapolation`, and
  `ExactNonEquilibriumExtrapolation` have direction-major boundary kernels;
  other `LBMBase` boundary conditions use the standard layout-adapting
  fallback. `ExtrapolationOutflowMultiphase` is not ported since
  `PallasBGK` is single-phase only. `ExtrapolationOutflow` is also not
  ported: its `indices_nbr` (core `boundary_conditions.py`) is not one
  neighbor per boundary node for a typical flat face (e.g. 256 boundary nodes
  vs. 512 "neighbors" on a 16x16 face), so a batched local kernel can't
  reproduce its existing global apply() correctly; it stays on the AoS
  fallback until that upstream mismatch is fixed.
- The neighbor-reading outflow kernels (`ConvectiveOutflow`,
  `NonEquilibriumExtrapolation`, `ExactNonEquilibriumExtrapolation`) exchange
  a 1-layer X halo only when a boundary node's neighbor actually crosses a
  shard boundary, so they stay correct for any number of GPUs without paying
  halo cost in the common case.
- Standard BGK exact-difference forcing is part of the Pallas collision.

Benchmark drivers and generated data remain in `/tmp` while designs are being
explored.
