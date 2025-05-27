#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Fields Computation from Volumetric Data


Given a vector field u(x) and a density filed rho(x) 
For each x compute

U(x) = ∫ dx' K(x,x') rho(x') u(x')  

where K(x,x') is a Gaussian kernel with lengthscale sigma_smooth

@author francescomori
"""

import argparse
import logging
import math
import zarr
from dask.diagnostics import ProgressBar
import dask.array as da
from dask.array.overlap import map_overlap
from scipy import ndimage




def parse_arguments():
    """
    Parse command-line arguments for eigenvalue analysis.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing:
            input-vector (str): Path to the input Zarr dataset containing the vector field.
            input-density (str): Path to the input Zarr dataset containing the density field.
            output (str): Path to save the output Zarr dataset containing the smoothed vector field
            sigma_smooth (float): Gaussian kernel sigma (in voxel length).
            depth (int|None): Overlap depth in voxels for smoothing; defaults to 3*sigma if None.
            verbose (bool): Enable INFO-level logging if True.
    """
    parser = argparse.ArgumentParser(
        description="Eigenvalue analysis of volumetric data structure tensor"
    )
    parser.add_argument(
        "-iv", "--input-vector",
        dest="input_vector",
        required=True,
        help="Path to the input Zarr dataset containing the vector field (e.g., vector.zarr)"
    )
    parser.add_argument(
        "-id", "--input-density",
        dest="input_density",
        required=True,
        help="Path to the input Zarr dataset containing the density field (e.g., density.zarr)"
    )
    parser.add_argument(
        "-o", "--output",
        default="local_field.zarr",
        help="Path to save the output Zarr dataset containing the smoothed vector field"
    )
    parser.add_argument(
        "-s", "--sigma-smooth",
        dest="sigma_smooth",
        type=float,
        default=2.0,
        help="Gaussian kernel sigma (in voxel length)"
    )
    parser.add_argument(
        "--overlap-depth",
        dest="depth",
        type=int,
        default=None,
        help="Overlap depth in voxels for smoothing; defaults to ceil(3*sigma) if None"
    )
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


def load_data(input_path: str) -> da.Array:
    """
    Load volumetric data from a Zarr store into a Dask array.

    Args:
        input_path (str): Path to the input Zarr dataset.
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
    return data


def smooth_data(data_vec: da.Array,
                data_dens: da.Array,
                sigma: float,
                depth: int = None) -> da.Array:
    """
    Compute U(x) = ∫ dx' K(x,x') rho(x') u(x') by Gaussian‐smoothing 
    the product rho(x') * u(x') for a channel‐last vector field.

    Args:
        data_vec (da.Array): Vector field.
        data_dens (da.Array): Density field.
        sigma (float): Gaussian kernel sigma in voxels.
        depth (int, optional): Overlap depth; defaults to ceil(3*sigma).

    Returns:
        da.Array: Smoothed weighted field, same shape as data_vec.
    """
    # determine overlap depth = 3σ if not provided
    if depth is None:
        depth = int(math.ceil(3 * sigma))

    
    # expand density onto the channel axis
    dens_expanded = data_dens[..., None]  # now shape (...,1) broadcasts to (...,3)

    # weight the vector by the density
    weighted = data_vec * dens_expanded

    # gaussian‐smooth across all spatial dimensions
    smoothed = map_overlap(
        lambda block: ndimage.gaussian_filter(block, sigma=sigma),
        weighted,
        depth={i: depth for i in range(weighted.ndim - 1)},  # only spatial axes
        boundary='reflect',
        trim=True
    )

    return smoothed


def save_results(eigenvalues: da.Array, output_path: str):
    """
    Save eigenvalues to a Zarr store at output_path.

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
    logging.info("Starting local fields computation with args: %s", args)

    data_vec = load_data(args.input_vector)
    logging.info("Vector field data loaded from %s", args.input_vector)
    
    data_den = load_data(args.input_density)
    logging.info("Density data loaded from %s", args.input_density)

    smoothed = smooth_data(data_vec,data_den, args.sigma_smooth, args.depth)
    logging.info("Smoothing complete")


    logging.info("Saving eigenvalues to %s", args.output)
    save_results(smoothed, args.output)




if __name__ == "__main__":
    main()


