"""Render Gaussian Splats with normal map computed from finite differences on depth.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from sharp.utils import vis

from sharp.utils import io, gsplat
from sharp.utils.gaussians import (
    Gaussians3D,
    save_ply,
    load_ply,
)

LOGGER = logging.getLogger(__name__)


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


def compute_normals_from_depth(depth, intrinsics):
    """
    Compute surface normals from depth map using finite differences.
    
    Args:
        depth: [H, W] depth map tensor
        intrinsics: [4, 4] camera intrinsics matrix
    
    Returns:
        normals: [H, W, 3] normal map in camera space
    """
    device = depth.device
    h, w = depth.shape
    
    f_x = intrinsics[0, 0].item()
    f_y = intrinsics[1, 1].item()
    c_x = intrinsics[0, 2].item()
    c_y = intrinsics[1, 2].item()
    
    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Backproject to 3D camera space
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    X = (x_coords - c_x) * depth / f_x
    Y = (y_coords - c_y) * depth / f_y
    Z = depth
    
    # Stack to get [H, W, 3]
    points_3d = torch.stack([X, Y, Z], dim=-1)
    
    # Compute gradients using finite differences
    # For each point, compute vectors to neighbors and cross product
    # Use central differences where possible, forward/backward at edges
    
    # Compute dx (gradient in x direction)
    dx = torch.zeros_like(points_3d)
    dx[:, 1:-1] = (points_3d[:, 2:] - points_3d[:, :-2]) / 2.0  # Central diff
    dx[:, 0] = points_3d[:, 1] - points_3d[:, 0]  # Forward diff
    dx[:, -1] = points_3d[:, -1] - points_3d[:, -2]  # Backward diff
    
    # Compute dy (gradient in y direction)
    dy = torch.zeros_like(points_3d)
    dy[1:-1, :] = (points_3d[2:, :] - points_3d[:-2, :]) / 2.0  # Central diff
    dy[0, :] = points_3d[1, :] - points_3d[0, :]  # Forward diff
    dy[-1, :] = points_3d[-1, :] - points_3d[-2, :]  # Backward diff
    
    # Normal is cross product of tangent vectors: n = dx × dy
    normals = torch.cross(dx, dy, dim=-1)
    
    # Normalize
    normals = F.normalize(normals, dim=-1)
    
    # Handle invalid normals (zero depth, nan, etc.)
    valid_mask = (depth > 0) & ~torch.isnan(depth) & ~torch.isnan(normals).any(dim=-1)
    normals[~valid_mask] = 0.0
    
    return normals


@click.command()
@click.option(
    "-i",
    "--input-ply",
    type=click.Path(path_type=Path, exists=True),
    help="Path to input .ply Gaussian Splat file.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save renderings.",
    required=True,
)
@click.option(
    "--azim",
    type=float,
    default=-35.0,
    help="Azimuth angle for camera in degrees.",
)
@click.option(
    "--elev",
    type=float,
    default=0.0,
    help="Elevation angle for camera in degrees.",
)
@click.option(
    "--dist",
    type=float,
    default=3.0,
    help="Distance from camera to object.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def render_normals_cli(
    input_ply: Path,
    output_path: Path,
    azim: float,
    elev: float,
    dist: float,
    verbose: bool,
):
    """Render Gaussian Splats with normals computed from depth using finite differences."""
    # Initialize logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True
    )
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {output_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")
    
    # Load Gaussian Splats
    LOGGER.info(f"Loading Gaussian Splats from {input_ply}")
    gaussians, metadata, _, _ = load_ply(input_ply)
    width, height = metadata.resolution_px
    f_px = metadata.focal_length_px
    
    LOGGER.info(f"Loaded {gaussians.mean_vectors[0].shape[0]} Gaussians")
    LOGGER.info(f"Resolution: {width}x{height}, focal length: {f_px:.2f}px")
    
    # Create intrinsics matrix
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
    
    # Compute object center for camera targeting
    gs_means = gaussians.mean_vectors[0].cpu().numpy()
    mean_pos = np.mean(gs_means, axis=0)
    bbox_min = np.min(gs_means, axis=0)
    bbox_max = np.max(gs_means, axis=0)
    
    LOGGER.info(f"Object center: {mean_pos}")
    LOGGER.info(f"Object bbox: min={bbox_min}, max={bbox_max}")
    
    # Setup camera view
    extrinsics = look_at_view_transform(
        dist=dist,
        elev=elev,
        azim=azim,
        at=mean_pos,
        up=[0.0, 1.0, 0.0]
    )
    
    LOGGER.info(f"Camera view: azim={azim:.1f}°, elev={elev:.1f}°, dist={dist:.2f}")
    
    # Setup renderer
    renderer = gsplat.GSplatRenderer(color_space="linear")
    
    # Render color and depth
    LOGGER.info("Rendering Gaussian Splats...")
    rendering_output = renderer(
        gaussians.to(device),
        extrinsics=extrinsics.to(device),
        intrinsics=intrinsics[None],
        image_width=width,
        image_height=height,
    )
    
    # Extract depth and color
    depth = rendering_output.depth[0].squeeze()  # [H, W]
    depth_np = depth.cpu().numpy()
    
    color = torch.clamp(rendering_output.color[0], 0.0, 1.0)
    color = (color.permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
    color_np = color.cpu().numpy()
    
    LOGGER.info(f"Depth stats: min={depth_np.min():.3f}, max={depth_np.max():.3f}, mean={depth_np.mean():.3f}")
    
    # Save rendered color
    stem = input_ply.stem
    io.save_image(color_np, output_path / f"{stem}_rendered.png")
    LOGGER.info(f"Saved rendered image")
    
    # Save depth visualization
    colored_depth_pt = vis.colorize_depth(
        depth.unsqueeze(0),
        min(depth_np.max(), vis.METRIC_DEPTH_MAX_CLAMP_METER),
    )
    colored_depth_np = colored_depth_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    io.save_image(colored_depth_np, output_path / f"{stem}_depth.png")
    LOGGER.info(f"Saved depth visualization")
    
    # Compute normals from depth using finite differences
    LOGGER.info("Computing normals from depth using finite differences...")
    normals = compute_normals_from_depth(depth, intrinsics)  # [H, W, 3]
    
    # Convert normals to color space for visualization (map [-1,1] to [0,1])
    normals_vis = (normals + 1.0) / 2.0
    normals_vis = torch.clamp(normals_vis, 0.0, 1.0)
    normals_vis = (normals_vis * 255.0).to(dtype=torch.uint8)
    normals_vis_np = normals_vis.cpu().numpy()
    
    io.save_image(normals_vis_np, output_path / f"{stem}_normal_fd.png")
    LOGGER.info(f"Saved finite difference normal map")
    
    # Also create a Gaussian Splat .ply file with normals as colors
    # Backproject depth to 3D points
    h, w = depth.shape
    f_x = intrinsics[0, 0].item()
    f_y = intrinsics[1, 1].item()
    c_x = intrinsics[0, 2].item()
    c_y = intrinsics[1, 2].item()
    
    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Backproject valid depth points
    valid_mask = (depth > 0) & ~torch.isnan(depth)
    valid_depth = depth[valid_mask]
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_normals = normals[valid_mask]  # [N, 3]
    
    # Camera space coordinates
    X_cam = (valid_x - c_x) * valid_depth / f_x
    Y_cam = (valid_y - c_y) * valid_depth / f_y
    Z_cam = valid_depth
    points_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=-1)  # [N, 3]
    
    # Transform to world space
    ext_np = extrinsics[0].cpu().numpy()
    ext_inv = np.linalg.inv(ext_np)
    points_cam_homo = torch.cat([points_cam, torch.ones((len(points_cam), 1), device=device)], dim=-1)
    points_world = (torch.from_numpy(ext_inv).to(device).float() @ points_cam_homo.T).T[:, :3]
    
    # Transform normals to world space (only rotation, no translation)
    R_inv = torch.from_numpy(ext_inv[:3, :3]).to(device).float()
    normals_world = (R_inv @ valid_normals.T).T
    normals_world = F.normalize(normals_world, dim=-1)
    
    # Create Gaussians with normals as colors
    num_points = len(points_world)
    LOGGER.info(f"Creating {num_points} point Gaussian Splat with normal colors")
    
    # Small uniform scales and identity rotations
    scales = torch.ones((num_points, 3), device=device) * 0.001
    quats = torch.zeros((num_points, 4), device=device)
    quats[:, 0] = 1.0  # w=1, x=y=z=0 (identity rotation)
    opacities = torch.ones((num_points,), device=device)
    normal_colors = (normals_world + 1.0) / 2.0  # Map to [0,1]
    
    gaussians_normals = Gaussians3D(
        mean_vectors=points_world.unsqueeze(0),
        singular_values=scales.unsqueeze(0),
        quaternions=quats.unsqueeze(0),
        colors=normal_colors.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
    )
    
    save_ply(gaussians_normals.to("cpu"), f_px, (height, width), output_path / f"{stem}_normal_fd_gs.ply")
    LOGGER.info(f"Saved normal Gaussian Splat file")
    
    LOGGER.info("Done!")


if __name__ == "__main__":
    render_normals_cli()
