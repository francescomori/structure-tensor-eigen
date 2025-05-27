# Volumetric Field Analysis Tools

This repository provides two command-line scripts for analyzing 3D volumetric data in Zarr format.

## eig_analysis.py

Compute per-voxel eigenvalues (and optional eigenvectors) of a 3D scalar field using the structure tensor.

### Features

* **Out-of-core processing:** leverages Dask `map_overlap` for volumes larger than memory
* **Flexible I/O:** reads and writes chunked Zarr datasets
* **Gaussian pre-smoothing:** optional smoothing of the input volume
* **Eigenpairs:** compute eigenvalues only, or both eigenvalues and eigenvectors
* **Voxel selection:** restrict to nonzero or all voxels

### Usage

```bash
# Eigenvalues only, smoothing σ=2.0
python eig_analysis.py \
  --input data_scalar.zarr \
  --output ev.zarr \
  --sigma-smooth 2.0 \
  --nonzero-only

# Eigenvalues + eigenvectors for all voxels
python eig_analysis.py \
  --input data_scalar.zarr \
  --output ev.zarr \
  --eigenvectors-output evectors.zarr \
  --sigma-smooth 2.0 \
  --all-voxels

# For full options:
python eig_analysis.py --help
```

## local\_fields.py

Compute density-weighted local vector fields by Gaussian smoothing of a density-modulated vector field.

### Features

* **Density weighting:** multiplies the vector field by the density before smoothing
* **Gaussian smoothing:** uses SciPy's `gaussian_filter` via Dask `map_overlap`
* **Chunk-safe:** configurable overlap depth (defaults to 3σ)

### Usage

```bash
# Basic density-weighted smoothing
python local_fields.py \
  --input-vector vector_field.zarr \
  --input-density density_field.zarr \
  --output smoothed_field.zarr \
  --sigma-smooth 2.5

# With custom overlap depth
python local_fields.py \
  --input-vector vector_field.zarr \
  --input-density density_field.zarr \
  --output smoothed_field.zarr \
  --sigma-smooth 1.5 \
  --overlap-depth 5

# For full options:
python local_fields.py --help
```
