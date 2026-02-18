"""Render Gaussian Splats from multiple camera viewpoints on a hemisphere.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
import json

import click
import numpy as np
import torch
import cv2 as cv
from sharp.utils import io, gsplat
from sharp.utils.gaussians import load_ply

LOGGER = logging.getLogger(__name__)


def look_at_view_transform(dist=1.0, elev=0.0, azim=0.0, degrees=True, at=None, up=None):
    """
    PyTorch3D-style look_at_view_transform function.
    Generates camera extrinsics from spherical coordinates.
    
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


def fibonacci_hemisphere(n_points, upper=True):
    """
    Generate evenly distributed points on hemisphere using Fibonacci lattice.
    
    Args:
        n_points: number of points to generate
        upper: if True, upper hemisphere (y >= 0); if False, lower hemisphere (y <= 0)
    
    Returns:
        Array of shape [n_points, 3] containing unit vectors on hemisphere
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
    
    # Generate points on hemisphere (upper or lower based on parameter)
    i = 0
    while len(points) < n_points:
        # Map i to [0, 2*n_points] range to cover full sphere, then filter
        y = 1 - (i / (2 * n_points - 1)) * 2  # y from 1 to -1
        
        # Filter based on hemisphere choice
        if (upper and y >= 0) or (not upper and y <= 0):
            radius_at_y = np.sqrt(1 - y * y)
            theta = phi * i
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            points.append([x, y, z])
        
        i += 1
        
        # Safety check to avoid infinite loop
        if i > 10 * n_points:
            LOGGER.warning(f"Could only generate {len(points)} points on hemisphere, needed {n_points}")
            break
    
    return np.array(points[:n_points], dtype=np.float32)


def cartesian_to_spherical(points):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        points: [N, 3] array of (x, y, z) unit vectors
    
    Returns:
        elevation: [N] array of elevation angles in degrees (0° = equator, 90° = north pole)
        azimuth: [N] array of azimuth angles in degrees (0° = +Z, 90° = +X, etc.)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Elevation: angle from XZ plane (equator) to point
    # y = sin(elev), where elev in [-90, 90]
    elev = np.arcsin(np.clip(y, -1.0, 1.0))  # radians
    
    # Azimuth: angle in XZ plane from +Z axis, rotating towards +X
    # x = cos(elev) * sin(azim)
    # z = cos(elev) * cos(azim)
    azim = np.arctan2(x, z)  # radians, range [-pi, pi]
    
    # Convert to degrees
    elev_deg = np.rad2deg(elev)
    azim_deg = np.rad2deg(azim)
    
    return elev_deg, azim_deg


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to .ply file containing Gaussian Splats.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered images.",
    required=True,
)
@click.option(
    "--n-cams",
    type=int,
    default=50,
    help="Number of camera viewpoints to render from (default: 50).",
)
@click.option(
    "--radius",
    type=float,
    default=3.0,
    help="Radius of hemisphere (distance from object center) in meters (default: 3.0).",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run on. ['cpu', 'cuda'] (default: 'cuda').",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
@click.option(
    "--lower-hemisphere",
    is_flag=True,
    default=False,
    help="Use lower hemisphere (y <= 0) instead of upper hemisphere (default: False).",
)
def render_multiview_cli(
    input_path: Path,
    output_path: Path,
    n_cams: int,
    radius: float,
    device: str,
    verbose: bool,
    lower_hemisphere: bool,
):
    """Render Gaussian Splats from multiple camera viewpoints on a hemisphere."""
    
    # Initialize logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True
    )
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"{timestamp}_multiview.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    LOGGER.info(f"Output directory: {output_path}")
    LOGGER.info(f"Logging to file: {log_file}")
    
    # Log configuration
    LOGGER.info("=== Configuration ===")
    LOGGER.info(f"input_path: {input_path}")
    LOGGER.info(f"output_path: {output_path}")
    LOGGER.info(f"n_cams: {n_cams}")
    LOGGER.info(f"radius: {radius}")
    LOGGER.info(f"device: {device}")
    LOGGER.info(f"verbose: {verbose}")
    LOGGER.info("====================")
    
    device = torch.device(device)
    
    # Load Gaussian Splats
    LOGGER.info(f"Loading Gaussian Splats from {input_path}")
    gaussians, metadata, _, _ = load_ply(input_path)
    width, height = metadata.resolution_px
    f_px = metadata.focal_length_px
    
    LOGGER.info(f"Loaded {gaussians.mean_vectors[0].shape[0]} Gaussians")
    LOGGER.info(f"Resolution: {width}x{height}, focal length: {f_px:.2f}px")
    
    # Compute intrinsics
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
    
    # Get object center
    gs_means = gaussians.mean_vectors[0].cpu().numpy()  # [N, 3]
    mean_pos = np.mean(gs_means, axis=0)
    std_pos = np.std(gs_means, axis=0)
    bbox_min = np.min(gs_means, axis=0)
    bbox_max = np.max(gs_means, axis=0)
    
    LOGGER.info(f"Object center: [{mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f}]")
    LOGGER.info(f"Object std: [{std_pos[0]:.3f}, {std_pos[1]:.3f}, {std_pos[2]:.3f}]")
    LOGGER.info(f"Object bbox: min={bbox_min}, max={bbox_max}")
    
    # Generate camera positions on hemisphere using Fibonacci lattice
    hemisphere_type = "lower" if lower_hemisphere else "upper"
    LOGGER.info(f"Generating {n_cams} camera positions on {hemisphere_type} hemisphere (radius={radius}m)...")
    unit_vectors = fibonacci_hemisphere(n_cams, upper=not lower_hemisphere)  # [N, 3] unit vectors on hemisphere
    
    # Convert to spherical coordinates
    elevations, azimuths = cartesian_to_spherical(unit_vectors)
    
    LOGGER.info(f"Elevation range: [{elevations.min():.1f}°, {elevations.max():.1f}°]")
    LOGGER.info(f"Azimuth range: [{azimuths.min():.1f}°, {azimuths.max():.1f}°]")
    
    # Compute actual camera positions in world space
    camera_positions = unit_vectors * radius + mean_pos  # [N, 3]
    
    # Save camera data to JSON for Three.js visualization
    camera_data = {
        "object_center": mean_pos.tolist(),
        "radius": float(radius),
        "n_cams": int(n_cams),
        "hemisphere": "lower" if lower_hemisphere else "upper",
        "cameras": [
            {
                "id": int(i + 1),
                "position": camera_positions[i].tolist(),
                "elevation": float(elevations[i]),
                "azimuth": float(azimuths[i])
            }
            for i in range(n_cams)
        ],
        "gs_ply_path": str(input_path.relative_to(output_path.parent) if input_path.is_relative_to(output_path.parent) else input_path)
    }
    
    camera_json_path = output_path / "camera_data.json"
    with open(camera_json_path, 'w') as f:
        json.dump(camera_data, f, indent=2)
    LOGGER.info(f"Saved camera data to {camera_json_path}")
    
    # Create visualization of camera locations
    LOGGER.info("Creating camera location visualization...")
    
    # Place visualization camera far away to see entire scene
    # Use a distance that's 2.5x the radius to get good view of hemisphere and object
    vis_radius = radius * 2.5
    vis_elev = 25.0  # Look from slightly above
    vis_azim = 45.0  # Angled view
    
    vis_extrinsics = look_at_view_transform(
        dist=vis_radius,
        elev=vis_elev,
        azim=vis_azim,
        at=mean_pos,
        up=[0.0, 1.0, 0.0]
    )
    
    # Render GS from visualization viewpoint
    renderer = gsplat.GSplatRenderer(color_space="linear")
    gaussians_gpu = gaussians.to(device)
    
    vis_rendering = renderer(
        gaussians_gpu,
        extrinsics=vis_extrinsics.to(device),
        intrinsics=intrinsics[None],
        image_width=width,
        image_height=height,
    )
    
    vis_color = torch.clamp(vis_rendering.color[0], 0.0, 1.0)
    vis_color = (vis_color.permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
    vis_color_np = vis_color.cpu().numpy().copy()  # Ensure contiguous array for OpenCV
    
    # Project camera positions to 2D image coordinates
    f_x = intrinsics[0, 0].item()
    f_y = intrinsics[1, 1].item()
    c_x = intrinsics[0, 2].item()
    c_y = intrinsics[1, 2].item()
    
    # Transform camera positions to visualization camera space
    cam_pos_homo = np.concatenate([camera_positions, np.ones((n_cams, 1))], axis=1)  # [N, 4]
    vis_ext_np = vis_extrinsics[0].cpu().numpy()
    cam_pos_vis = (vis_ext_np @ cam_pos_homo.T).T  # [N, 4]
    
    # Project to 2D
    valid_projections = []
    for i in range(n_cams):
        X, Y, Z = cam_pos_vis[i, :3]
        if Z > 0:  # Camera is in front of visualization camera
            u = int(round((f_x * X / Z) + c_x))
            v = int(round((f_y * Y / Z) + c_y))
            if 0 <= u < width and 0 <= v < height:
                valid_projections.append((u, v, i))
    
    LOGGER.info(f"Projecting {len(valid_projections)}/{n_cams} camera positions onto visualization")
    
    # Draw red circles at camera locations with ID labels
    for u, v, cam_idx in valid_projections:
        cv.circle(vis_color_np, (u, v), radius=8, color=(255, 0, 0), thickness=-1)  # Filled red circle
        cv.circle(vis_color_np, (u, v), radius=10, color=(255, 255, 255), thickness=2)  # White outline
        
        # Add camera ID label
        label = f"{cam_idx+1}"
        # Position text slightly offset from circle
        text_x = u + 15
        text_y = v + 5
        cv.putText(vis_color_np, label, (text_x, text_y), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)  # White text with thick outline
        cv.putText(vis_color_np, label, (text_x, text_y), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)  # Red text on top
    
    # Draw object center as a green cross
    obj_center_homo = np.append(mean_pos, 1.0)
    obj_center_vis = vis_ext_np @ obj_center_homo
    if obj_center_vis[2] > 0:
        u_obj = int(round((f_x * obj_center_vis[0] / obj_center_vis[2]) + c_x))
        v_obj = int(round((f_y * obj_center_vis[1] / obj_center_vis[2]) + c_y))
        if 0 <= u_obj < width and 0 <= v_obj < height:
            cv.drawMarker(vis_color_np, (u_obj, v_obj), color=(0, 255, 0), 
                         markerType=cv.MARKER_CROSS, markerSize=20, thickness=3)
    
    # Save visualization
    cam_locations_path = output_path / "cam_locations.png"
    io.save_image(vis_color_np, cam_locations_path)
    LOGGER.info(f"Saved camera location visualization to {cam_locations_path}")
    
    # Setup renderer for actual multi-view rendering
    renderer = gsplat.GSplatRenderer(color_space="linear")
    gaussians_gpu = gaussians.to(device)
    
    # Render from each camera position
    LOGGER.info("Starting rendering...")
    for cam_idx in range(n_cams):
        elev = elevations[cam_idx]
        azim = azimuths[cam_idx]
        
        # Generate camera extrinsics
        extrinsics = look_at_view_transform(
            dist=radius,
            elev=elev,
            azim=azim,
            at=mean_pos,
            up=[0.0, 1.0, 0.0]  # Y-axis up
        )
        
        # Compute actual camera position for logging
        cam_pos = unit_vectors[cam_idx] * radius + mean_pos
        
        if cam_idx % 10 == 0 or cam_idx < 5:
            LOGGER.info(f"Camera {cam_idx+1}/{n_cams}: elev={elev:.1f}°, azim={azim:.1f}°, pos=[{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
        
        # Render
        rendering_output = renderer(
            gaussians_gpu,
            extrinsics=extrinsics.to(device),
            intrinsics=intrinsics[None],
            image_width=width,
            image_height=height,
        )
        
        # Extract and save color image
        color = torch.clamp(rendering_output.color[0], 0.0, 1.0)
        color = (color.permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        color_np = color.cpu().numpy()
        
        # Save as cam01.png, cam02.png, etc.
        output_filename = output_path / f"cam{cam_idx+1:02d}.png"
        io.save_image(color_np, output_filename)
    
    LOGGER.info(f"Rendering complete! Saved {n_cams} images to {output_path}")
    LOGGER.info(f"Images named: cam01.png to cam{n_cams:02d}.png")


if __name__ == "__main__":
    render_multiview_cli()
