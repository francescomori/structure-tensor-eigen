# Structure‐Tensor Eigen Analysis

Compute per‐voxel eigenvalues (and eigenvectors) of the 3D structure tensor from volumetric Zarr data, using Dask for out‐of‐core performance.

## Features

- **Out‐of-core processing:** leverages Dask arrays and map_overlap for large volumes  
- **Flexible I/O:** reads/writes chunked Zarr datasets  
- **Eigenpairs:** optionally compute both eigenvalues and eigenvectors  

## Usage

```bash
# eigenvalues only
eig-analysis.py \
  --input data.zarr \
  --output ev.zarr \
  --sigma-smooth 2.0 \
  --nonzero-only

# eigenvalues + eigenvectors
eig-analysis.py \
  --input data.zarr \
  --output ev.zarr \
  --eigenvectors-output evectors.zarr \
  --sigma-smooth 2.0 \
  --all-voxels
```

For full options, run:
```bash
eig-analysis --help
```