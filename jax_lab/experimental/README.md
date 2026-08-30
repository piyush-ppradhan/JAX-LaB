# Experimental Pallas backend
GPU-optimized SoA implementations of single- and multiphase BGK/MRT:
- Singlephase: `PallasBGK`, `PallasMRT`
- Multiphase: `PallasMultiphaseBGK`, `PallasMultiphaseMRT`.

## Implementation
- Preserves standard JAX-LaB interface, but remains experimental. 
- Unsupported configurations raise an error or use the standard boundary-condition fallback.

## Support
- Lattice types: D2Q9, D3Q19, D3Q27
- Precision: mixed precision
- Boundary conditions: SoA boundary conditions 
- Forcing scheme: SoA implementation of exact difference method (EDM)
- Multi-GPU: yes
