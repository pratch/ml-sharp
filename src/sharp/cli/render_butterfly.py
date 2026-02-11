# Rule: tmux send -t 1 '!dpython' Enter Enter
"""Render butterfly Gaussian Splat from multiple azimuth angles.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch

from sharp.utils import io, gsplat
from sharp.utils.gaussians import load_ply

LOGGER = logging.getLogger(__name__)

#hardcoded input file name
input_file_name = "butterfly"


def look_at_view_transform(dist=1.0, elev=0.0, azim=0.0, degrees=True, at=None, up=None):
    """
    PyTorch3D-style look_at_view_transform function.
    Generates camera extrinsics from spherical coordinates or direct specification.
    
    Args:
        dist: distance from camera to the object (float or array)
        elev: elevation angle (float or array)
        azim: azimuth angle (float or array)
        degrees: if True, angles are in degrees; if False, in radians
        at: the point(s) to look at, shape [3] or [N, 3], defaults to origin
        up: the up direction, shape [3] or [N, 3], defaults to [0, 1, 0]
    
    Returns:
        4x4 extrinsics matrix as torch tensor [1, 4, 4] or [N, 4, 4]
    """
    # Handle defaults
    if at is None:
        at = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    else:
        at = np.array(at, dtype=np.float32)
        if at.ndim == 1:
            at = at[np.newaxis, :]
    
    if up is None:
        up = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    else:
        up = np.array(up, dtype=np.float32)
        if up.ndim == 1:
            up = up[np.newaxis, :]
    
    # Convert scalars to arrays
    dist = np.atleast_1d(dist).astype(np.float32)
    elev = np.atleast_1d(elev).astype(np.float32)
    azim = np.atleast_1d(azim).astype(np.float32)
    
    # Convert to radians if needed
    if degrees:
        elev = np.deg2rad(elev)
        azim = np.deg2rad(azim)
    
    # Compute camera position from spherical coordinates
    # Convention: azim rotates around Y axis, elev rotates around X axis
    x = dist * np.cos(elev) * np.sin(azim)
    y = dist * np.sin(elev)
    z = dist * np.cos(elev) * np.cos(azim)
    
    camera_position = np.stack([x, y, z], axis=-1)  # [N, 3]
    
    # Add 'at' offset to camera position
    camera_position = camera_position + at
    
    # Build extrinsics matrices for each camera
    batch_size = len(dist)
    extrinsics_list = []
    
    for i in range(batch_size):
        eye = camera_position[i]
        target = at[i % len(at)]
        up_vec = up[i % len(up)]
        
        # Compute camera coordinate frame
        z_axis = target - eye  # Forward (looking direction)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(z_axis, up_vec)  # Right
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        
        y_axis = np.cross(x_axis, z_axis)  # Up
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        
        # Build rotation matrix (world-to-camera)
        # For OpenCV convention: camera looks down +Z axis
        R = np.stack([x_axis, y_axis, z_axis], axis=0)  # [3, 3]
        
        # Translation
        t = -R @ eye  # [3]
        
        # Build 4x4 extrinsics matrix
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t
        
        extrinsics_list.append(extrinsics)
    
    extrinsics_array = np.stack(extrinsics_list, axis=0)  # [N, 4, 4]
    return torch.from_numpy(extrinsics_array)


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def render_cli(verbose: bool):
    # Initialize logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True
    )
    
    device = torch.device("cuda")

    # Hardcoded paths
    input_ply = Path(f"output/{input_file_name}.ply")
    output_dir = Path("output/renders")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Loading Gaussians from {input_ply}")
    gaussians, metadata, _, _ = load_ply(input_ply)
    width, height = metadata.resolution_px
    f_px = metadata.focal_length_px
    
    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2.0, 0],
            [0, f_px, (height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )

    # Get object center for camera targeting
    gs_means = gaussians.mean_vectors[0].cpu().numpy()  # [N, 3]
    mean_pos = np.mean(gs_means, axis=0)
    bbox_min = np.min(gs_means, axis=0)
    bbox_max = np.max(gs_means, axis=0)
    LOGGER.info(f"Gaussian center: {mean_pos}")
    LOGGER.info(f"Gaussian bbox: min={bbox_min}, max={bbox_max}")
    
    # Setup renderer
    renderer = gsplat.GSplatRenderer(color_space="linear")

    # Render from multiple azimuth angles
    azim_range = range(-90, 91, 10)  # -90 to 90 with step 10
    LOGGER.info(f"Rendering from {len(azim_range)} azimuth angles: {list(azim_range)}")
    
    for idx, azim in enumerate(azim_range):
        ext_id_azim = look_at_view_transform(
            dist=3.0,
            elev=0.0,
            azim=float(azim),
            at=mean_pos,
            up=[0.0, 1.0, 0.0]
        )
        
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=ext_id_azim.to(device),
            intrinsics=intrinsics[None],
            image_width=width,
            image_height=height,
        )
        
        # Clamp to [0, 1] before converting to uint8 to prevent overflow artifacts
        color = torch.clamp(rendering_output.color[0], 0.0, 1.0)
        color = (color.permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        color = color.cpu().numpy()
        
        output_filename = output_dir / f"{input_file_name}_rendered_{idx:02d}.png"
        io.save_image(color, output_filename)
        LOGGER.info(f"Saved render {idx:02d} (azim={azim}°) to {output_filename}")
    
    LOGGER.info(f"Rendering complete! Saved {len(azim_range)} images to {output_dir}")


if __name__ == "__main__":
    render_cli()
