#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigenvalue and eigenvector Analysis of the Structure Tensor of Volumetric Data

General Notes:
    - Returns eigenvalues in ascending order.
    - The eigenvector matrix `v` has its i-th column (`v[:, i]`) as the eigenvector for eigenvalue `w[i]`.

@author francescomori
"""

import argparse
import logging
import math
import numpy as np
import zarr
from dask.diagnostics import ProgressBar
import dask.array as da
from dask.array.overlap import map_overlap
from scipy import ndimage
from dask import persist



def eigvals3(a):
    """Compute sorted eigenvalues of a symmetric 3×3 array."""
    return np.linalg.eigvalsh(a)


def eig3(a):
    """
    Compute eigenvalues AND eigenvectors for a symmetric 3×3 matrix.
    Returns:
      combined: 1D array of length 12, where
        combined[0:3] are the eigenvalues,
        combined[3:12] are the eigenvectors flattened.
    """
    w, v = np.linalg.eigh(a)
    combined = np.concatenate([w, v.ravel()]) 
    return combined

def parse_arguments():
    """
    Parse command-line arguments for eigenvalue analysis.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing:
            input (str): Path to the input Zarr file.
            output (str): Path to the output Zarr file.
            eigenvectors-output (str|None) : Path to output Zarr store for eigenvectors.
                                            If set, both eigenvalues and eigenvectors will be computed.
            sigma_smooth (float): Gaussian smoothing sigma (in voxel length).
            depth (int|None): Overlap depth in voxels for smoothing; defaults to 3*sigma if None.
            nonzero_only (bool): Flag to compute eigenvalues only on nonzero regions.
            verbose (bool): Enable INFO-level logging if True.
    """
    parser = argparse.ArgumentParser(
        description="Eigenvalue analysis of volumetric data structure tensor"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input Zarr dataset (e.g., data.zarr)"
    )
    parser.add_argument(
        "-o", "--output",
        default="eig.zarr",
        help="Path to save the output Zarr dataset"
    )
    parser.add_argument(
        "-e", "--eigenvectors-output", dest="eigenvectors_output", default=None,
        help="Path to output Zarr store for eigenvectors. "
             "If set, both eigenvalues and eigenvectors will be computed."
    )
    parser.add_argument(
        "-s", "--sigma-smooth",
        dest="sigma_smooth",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma (in voxel length)"
    )
    parser.add_argument(
        "--overlap-depth",
        dest="depth",
        type=int,
        default=None,
        help="Overlap depth in voxels for smoothing; defaults to ceil(3*sigma) if None"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--nonzero-only",
        dest="nonzero_only",
        action="store_true",
        help="Sets to zero the output in regions where the input is zero"
    )
    group.add_argument(
        "--all-voxels",
        dest="nonzero_only",
        action="store_false",
        help="Compute eigenvalues on all voxels (overrides --nonzero-only)"
    )
    parser.set_defaults(nonzero_only=True)
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (INFO-level) logging"
    )
    return parser.parse_args()


def setup_logging(verbose: bool):
    """
    Configure the root logger.

    Args:
        verbose (bool): If True, set level to INFO; otherwise WARNING.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )


def load_data(input_path: str, nonzero_only: bool) -> da.Array:
    """
    Load volumetric data from a Zarr store into a Dask array.

    Args:
        input_path (str): Path to the input Zarr dataset.
        nonzero_only (bool): If True, downstream steps should skip zero-valued voxels.

    Returns:
        da.Array: Dask array representing the volumetric data.
    """
    logging.info("Opening Zarr store at %s", input_path)
    try:
        arr = zarr.open_array(input_path, mode='r')
    except Exception as e:
        logging.error("Failed to open Zarr array: %s", e)
        raise
    data = da.from_array(arr, chunks=arr.chunks)

    if nonzero_only:
        logging.info("Zero-valued voxels will be skipped during analysis (nonzero_only=True)")
    return data


def smooth_data(data: da.Array, sigma: float, depth: int = None) -> da.Array:
    """
    Apply Gaussian smoothing to the data using Dask's map_overlap.

    Args:
        data (da.Array): Input volumetric data.
        sigma (float): Gaussian smoothing sigma (in voxel length).
        depth (int, optional): Overlap depth in voxels. If None, defaults to ceil(3*sigma).

    Returns:
        da.Array: Smoothed data array.
    """
    if depth is None:
        depth = int(math.ceil(3 * sigma))
    logging.info("Smoothing with sigma=%.2f and overlap depth=%d", sigma, depth)
    return map_overlap(
        lambda x: ndimage.gaussian_filter(x, sigma),
        data,
        depth=depth,
        boundary='reflect',
        trim=True
    )


def compute_structure_tensor(data: da.Array, sigma: float, depth: int = None) -> da.Array:
    """
    Compute and smooth the six unique components of the 3D structure tensor for each voxel.

    Args:
        data (da.Array): Smoothed volumetric data.
        sigma (float): Gaussian smoothing sigma for tensor components.
        depth (int, optional): Overlap depth in voxels for component smoothing; if None, defaults to ceil(3*sigma).

    Returns:
        da.Array: Dask array of shape (6, z, y, x) containing the smoothed tensor components.
    """
    logging.info("Computing gradients for structure tensor...")
    grad_depth = {0: 1, 1: 1, 2: 1}
    boundary = 'reflect'
    Gz = map_overlap(
        lambda arr: ndimage.sobel(arr, axis=0),
        data,
        depth=grad_depth,
        boundary=boundary,
        trim=True
    )
    Gy = map_overlap(
        lambda arr: ndimage.sobel(arr, axis=1),
        data,
        depth=grad_depth,
        boundary=boundary,
        trim=True
    )
    Gx = map_overlap(
        lambda arr: ndimage.sobel(arr, axis=2),
        data,
        depth=grad_depth,
        boundary=boundary,
        trim=True
    )
    logging.info("Forming raw tensor components...")
    comps = [Gx * Gx, Gx * Gy, Gx * Gz, Gy * Gy, Gy * Gz, Gz * Gz]

    if depth is None:
        depth = int(math.ceil(3 * sigma))
    logging.info("Smoothing tensor components with sigma=%.2f and overlap depth=%d", sigma, depth)
    smoothed = [
        map_overlap(
            lambda x: ndimage.gaussian_filter(x, sigma),
            comp,
            depth=depth,
            boundary='reflect',
            trim=True
        ) for comp in comps
    ]
    return da.stack(smoothed, axis=0)


def extract_eigenvalues(tensor_components: da.Array) -> da.Array:
    """
    Extract eigenvalues from the 3D structure tensor at each voxel using apply_gufunc.

    Args:
        tensor_components (da.Array): Array of shape (6, z, y, x) with tensor components.

    Returns:
        da.Array: Eigenvalues array of shape (z, y, x, 3) sorted ascending.
    """
    logging.info("Extracting eigenvalues with apply_gufunc...")
    Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = tensor_components
    T = da.stack([
        da.stack([Jxx, Jxy, Jxz], axis=-1),
        da.stack([Jxy, Jyy, Jyz], axis=-1),
        da.stack([Jxz, Jyz, Jzz], axis=-1)
    ], axis=-2)
    eig_vs = da.apply_gufunc(
        eigvals3,
        "(i,j)->(i)",
        T,
        vectorize=True,
        output_dtypes=float,
        allow_rechunk=True
    )  # shape (z, y, x, 3)
    return eig_vs

def extract_eigenpairs(tensor_components: da.Array):
    """
    Extract both eigenvalues and eigenvectors at each voxel.
    Returns two Dask arrays:
      w: shape (...,3)
      v: shape (...,3,3)
    """
    Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = tensor_components
    T = da.stack([
        da.stack([Jxx, Jxy, Jxz], axis=-1),
        da.stack([Jxy, Jyy, Jyz], axis=-1),
        da.stack([Jxz, Jyz, Jzz], axis=-1)
    ], axis=-2)
    
    eig_vs = da.apply_gufunc(
    eig3,
    "(i,j)->(k)",       # <-- signature as a positional arg
    T,                  # <-- now the third positional arg
    output_sizes={"k": 12},
    vectorize=True,
    output_dtypes=float,
    allow_rechunk=True,
)# shape (z, y, x, 12)
    
    w = eig_vs[..., :3]            # shape (..., 3)
    
      
    row0 = eig_vs[...,   3:6]        
    row1 = eig_vs[...,   6:9]
    row2 = eig_vs[...,   9:]
    v = da.stack([row0, row1, row2], axis=-2)  # shape (..., 3, 3)
    return w,v


def save_results(eigenvalues: da.Array, output_path: str):
    """
    Save eigenvalues to a Zarr store at output_path with a clean progress bar.

    Args:
        eigenvalues (da.Array): Eigenvalues of shape (z, y, x, 3).
        output_path (str): Path to the output Zarr store.
    """
    logging.info("Writing to %s", output_path)
    with ProgressBar():
        eigenvalues.to_zarr(output_path, overwrite=True)


def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    logging.info("Starting eigenvalue analysis with args: %s", args)

    data = load_data(args.input, args.nonzero_only)
    logging.info("Data loaded from %s", args.input)

    smoothed = smooth_data(data, args.sigma_smooth, args.depth)
    logging.info("Smoothing complete")

    tensor = compute_structure_tensor(smoothed, args.sigma_smooth, args.depth)
    logging.info("Structure tensor computed (lazy)")

    if args.eigenvectors_output:
       logging.info("Extracting eigenvalues AND eigenvectors")
       
       w_lazy, v_lazy = extract_eigenpairs(tensor)
       
       if args.nonzero_only:
           logging.info("Masking results outside non-zero regions")
           mask = data != 0
           w_lazy  = da.where(mask[...,      None], w_lazy,  0)
           v_lazy = da.where(mask[..., None, None], v_lazy, 0)
        
       with ProgressBar():
            eigenvalues, eigenvectors = persist(w_lazy, v_lazy)
       
       logging.info("Eigenpair extraction complete")

       

       logging.info("Saving eigenvalues to %s", args.output)
       save_results(eigenvalues, args.output)
       logging.info("Saving eigenvectors to %s", args.eigenvectors_output)
       save_results(eigenvectors, args.eigenvectors_output)

    else:
       logging.info("Extracting eigenvalues only")
       eigenvalues = extract_eigenvalues(tensor)
       logging.info("Eigenvalue extraction complete")

       if args.nonzero_only:
           logging.info("Masking eigenvalues outside non-zero regions")
           mask = data != 0
           eigenvalues = da.where(mask[..., None], eigenvalues, 0)

       logging.info("Saving eigenvalues to %s", args.output)
       save_results(eigenvalues, args.output)




if __name__ == "__main__":
    main()


