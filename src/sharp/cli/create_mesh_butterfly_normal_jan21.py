# Rule: tmux send -t 1 '!dpython' Enter Enter
"""Contains `sharp predict` CLI implementation.

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
import torch.utils.data
import json
from sharp.utils import vis

from sharp.models import (
    PredictorParams,
    # RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io, gsplat
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    load_ply,
    unproject_gaussians,
)

from sharp.cli.render import render_gaussians
import cv2 as cv
from PIL import Image
from collections import deque
from scipy.spatial import KDTree

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


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
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default="sharp_2572gikvuh.pt",
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    verbose: bool,
):
    # Initialize logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True
    )
    
    device = torch.device("cuda")

    extensions = io.get_supported_image_extensions()
    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    for image_path in image_paths:
        # if "441902" not in str(image_path):
        if "butterfly" not in str(image_path):
            continue
        # output_depth_path = output_path / f"{image_path.stem}.npy"

        # Check if mesh outputs already exist
        mesh_output_path = output_path / f"{image_path.stem}_mesh.ply"
        mesh_pruned_output_path = output_path / f"{image_path.stem}_pruned.ply"
        glb_output_path = output_path / f"{image_path.stem}_mesh.glb"
        
        if mesh_output_path.exists() and mesh_pruned_output_path.exists():
            LOGGER.info("Mesh files for %s already exist, skipping.", image_path.stem)
            continue

        gaussians, metadata, _, _ = load_ply(output_path / f"{image_path.stem}.ply")
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
        std_pos = np.std(gs_means, axis=0)
        bbox_min = np.min(gs_means, axis=0)
        bbox_max = np.max(gs_means, axis=0)
        LOGGER.info(f"Gaussian center: {mean_pos}")
        LOGGER.info(f"Gaussian std: {std_pos}")
        LOGGER.info(f"Gaussian bbox: min={bbox_min}, max={bbox_max}")
        
        # PyTorch3D-style camera setup using spherical coordinates
        # For butterfly at y=offset with wings in X-Z plane, use Z as up
        ext_id = look_at_view_transform(
            dist=3.0,           # Distance from object
            elev=0.0,           # Elevation angle in degrees
            azim=90.0,          # Rotate 90° to look along -Y axis
            at=mean_pos,        # Look at object center
            up=[0.0, 1.0, 0.0]  # Y is up for butterfly
        )
        
        # Option 2: For mlsharp at z=offset (uncomment to use):
        # ext_id = look_at_view_transform(
        #     dist=3.0,
        #     elev=0.0,
        #     azim=0.0,
        #     at=mean_pos,
        #     up=[0.0, 1.0, 0.0]  # Y is up
        # )
        
        # Debug: compute and log actual camera position
        cam_x = 3.0 * np.cos(np.deg2rad(0.0)) * np.sin(np.deg2rad(90.0)) + mean_pos[0]
        cam_y = 3.0 * np.sin(np.deg2rad(0.0)) + mean_pos[1]
        cam_z = 3.0 * np.cos(np.deg2rad(0.0)) * np.cos(np.deg2rad(90.0)) + mean_pos[2]
        LOGGER.info(f"Camera position: [{cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f}]")
        LOGGER.info(f"Camera looking at: {mean_pos}")
        LOGGER.info(f"Camera up vector: [0, 0, 1]")
        LOGGER.info(f"Extrinsics matrix:\n{ext_id[0]}")
        
        # Debug: Transform some Gaussians to camera space to check if they're visible
        gs_means_homo = np.concatenate([gs_means, np.ones((len(gs_means), 1))], axis=1)  # [N, 4]
        ext_np = ext_id[0].numpy()
        gs_means_cam = (ext_np @ gs_means_homo.T).T  # [N, 4]
        depths_cam = gs_means_cam[:, 2]  # Z coordinate in camera space
        LOGGER.info(f"Gaussians in camera space - Z (depth) stats: min={depths_cam.min():.3f}, max={depths_cam.max():.3f}, mean={depths_cam.mean():.3f}")
        LOGGER.info(f"Number of Gaussians with positive depth (in front): {(depths_cam > 0).sum()} / {len(depths_cam)}")

        # Setup renderer for mesh generation
        renderer = gsplat.GSplatRenderer(color_space="linear")
        
        # Single view rendering for mesh generation
        ext_id = look_at_view_transform(
            dist=3.0,
            elev=0.0,
            azim=-50.0,  # Adjust azimuth as needed for mesh generation
            at=mean_pos,
            up=[0.0, 1.0, 0.0]
        )
        
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=ext_id.to(device),
            intrinsics=intrinsics[None],
            image_width=width,
            image_height=height,
        )

        # Extract depth and color
        depth = rendering_output.depth[0]
        depth_np = depth.cpu().numpy()
        color = torch.clamp(rendering_output.color[0], 0.0, 1.0)
        color = (color.permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        color = color.cpu().numpy()
        
        LOGGER.info(f"Depth stats: shape={depth_np.shape}, min={depth_np.min():.3f}, max={depth_np.max():.3f}")
        
        # Save rendered view
        io.save_image(color, output_path / f"{image_path.stem}_rendered.png")
        
        # Save depth visualization (same as predict_with_info.py)
        colored_depth_pt = vis.colorize_depth(
            depth,
            min(depth_np.max(), vis.METRIC_DEPTH_MAX_CLAMP_METER),
        )
        colored_depth_np = colored_depth_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
        io.save_image(colored_depth_np, output_path / f"{image_path.stem}_rawdepth.png")
        LOGGER.info(f"Saved depth visualization to {output_path / f'{image_path.stem}_rawdepth.png'}")

        # Compute and render normal map
        quats = gaussians.quaternions[0].to(device)  # [N, 4]
        scales = gaussians.singular_values[0].to(device)  # [N, 3]
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        # Rotation matrix from quaternion
        R = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1),
        ], dim=-2)  # [N, 3, 3]
        
        # Find the axis with smallest scale (the normal direction for flat Gaussians)
        min_scale_idx = torch.argmin(scales, dim=1)  # [N]
        
        # Extract the corresponding column from rotation matrix
        normals = torch.zeros((R.shape[0], 3), device=device)
        for i in range(3):
            mask = (min_scale_idx == i)
            normals[mask] = R[mask, :, i]

        # force random normals for testing
        # normals = torch.randn_like(normals)
        
        normals = F.normalize(normals, dim=-1)
        
        # Log scale statistics for debugging
        LOGGER.info(f"Scale statistics: min={scales.min().item():.6f}, max={scales.max().item():.6f}, mean={scales.mean().item():.6f}")
        LOGGER.info(f"Scale ratios (max/min per Gaussian): mean={((scales.max(dim=1)[0] / (scales.min(dim=1)[0] + 1e-8)).mean().item()):.2f}")
        
        # Create a copy of gaussians with normals as colors for rendering
        gaussians_with_normals = Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=gaussians.singular_values,
            quaternions=gaussians.quaternions,
            colors=((normals + 1.0) / 2.0).cpu().unsqueeze(0),  # Map [-1,1] to [0,1] for visualization
            opacities=gaussians.opacities,
        )
        
        # Save gaussians with normals as colors (move to CPU first)
        save_ply(gaussians_with_normals.to("cpu"), f_px, (height, width), output_path / f"{image_path.stem}_normals_gs.ply")
        LOGGER.info(f"Saved normal Gaussians to {output_path / f'{image_path.stem}_normals_gs.ply'}")
        
        normal_rendering = renderer(
            gaussians_with_normals.to(device),
            extrinsics=ext_id.to(device),
            intrinsics=intrinsics[None],
            image_width=width,
            image_height=height,
        )
        
        normal_map = (normal_rendering.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        normal_map = normal_map.cpu().numpy()
        io.save_image(normal_map, output_path / f"{image_path.stem}_normals.png")
        LOGGER.info(f"Saved normal map to {output_path / f'{image_path.stem}_normals.png'}")

        # Apply gaussian blur to the depth (squeeze to 2D first)
        depth_2d = depth_np.squeeze()  # Remove singleton dimensions
        depth_blurred = cv.GaussianBlur(depth_2d, (5, 5), 0)
        depth_blurred = cv.GaussianBlur(depth_blurred, (5, 5), 0)
        depth_blurred = cv.GaussianBlur(depth_blurred, (5, 5), 0)

        h, w = depth_2d.shape[0], depth_2d.shape[1]
        f_x = intrinsics[0, 0].item()
        f_y = intrinsics[1, 1].item()
        c_x = intrinsics[0, 2].item()
        c_y = intrinsics[1, 2].item()

        # Segment Gaussians in 3D using BFS
        segments = segment_gaussians_3d_bfs(
            gaussians,
            device,
            k_neighbors=10,
            normal_threshold=0.5,  # Relaxed: allow more normal variation
            distance_threshold=0.5,  # Increased from 0.1 to 0.5 meters
            color_threshold=0.5,  # Relaxed: allow more color variation
        )

        # Process each segment to create a mesh
        for seg_idx, segment_indices in enumerate(segments):
            LOGGER.info(f"Processing segment {seg_idx} with {len(segment_indices)} Gaussians")
            
            # Create mask for this segment by projecting GS points to 2D
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Get segment Gaussian means
            segment_means = gaussians.mean_vectors[0][segment_indices].cpu().numpy()  # [M, 3]
            
            # Project each Gaussian to image plane and mark in mask
            for mean_3d in segment_means:
                X, Y, Z = mean_3d
                if Z <= 0:
                    continue
                u = int(round((f_x * X / Z) + c_x))
                v = int(round((f_y * Y / Z) + c_y))
                if 0 <= v < h and 0 <= u < w:
                    mask[v, u] = 1
            
            # Dilate mask to fill small gaps
            kernel = np.ones((5, 5), np.uint8)
            mask = cv.dilate(mask, kernel, iterations=3)
            mask = cv.erode(mask, kernel, iterations=2)

            cv.imwrite(output_path / f"{image_path.stem}_seg{seg_idx}_mask.png", mask * 255)

            # Extract masked depth and color and compute vertices
            effective_mask = mask
            rows, cols = np.where(effective_mask == 1)

            if len(rows) == 0:
                LOGGER.warning(f"No masked pixels found for segment {seg_idx}. Skipping.")
                continue

            # Extract depth and color values for the masked pixels
            masked_depth_values = depth_blurred[rows, cols]
            masked_color_values = color[rows, cols]

            vertices = []
            vertex_colors = []
            vertex_map = {}

            vertex_idx_counter = 0
            for i in range(len(rows)):
                r, c = rows[i], cols[i]
                d = masked_depth_values[i]

                # Skip invalid or zero depth values
                if np.isnan(d) or d <= 0:
                    continue

                # Unproject 2D pixel (c, r) with depth d to 3D point (x, y, z) in camera coordinates
                x = (c - c_x) * d / f_x
                y = (r - c_y) * d / f_y
                z = d

                vertices.append([x, y, z])
                vertex_colors.append(masked_color_values[i].tolist())
                vertex_map[(r, c)] = vertex_idx_counter
                vertex_idx_counter += 1

            if not vertices:
                LOGGER.warning(f"No valid 3D points for segment {seg_idx}. Skipping.")
                continue

            LOGGER.info(f"Segment {seg_idx}: {len(vertices)} vertices generated.")
            vertices_np = np.array(vertices, dtype=np.float32)
            vertex_colors_np = np.array(vertex_colors, dtype=np.uint8)

            # 2. Triangulation: Create faces by connecting 2x2 blocks of masked pixels
            faces = []
            for r in range(h - 1):
                for c in range(w - 1): 
                    # Check if all four corners of the 2x2 block are within the masked region
                    if (
                        effective_mask[r, c]
                        and effective_mask[r + 1, c]
                        and effective_mask[r, c + 1]
                        and effective_mask[r + 1, c + 1]
                    ):
                        # Get the vertex indices for the four corners from the vertex_map
                        idx_rc = vertex_map.get((r, c))
                        idx_r1c = vertex_map.get((r + 1, c))
                        idx_rc1 = vertex_map.get((r, c + 1))
                        idx_r1c1 = vertex_map.get((r + 1, c + 1))

                        # Ensure all four corner vertices were valid
                        if all(idx is not None for idx in [idx_rc, idx_r1c, idx_rc1, idx_r1c1]):
                            # Create two triangles from the quadrilateral
                            faces.append([idx_rc, idx_r1c, idx_rc1])
                            faces.append([idx_r1c, idx_r1c1, idx_rc1])

            if not faces:
                LOGGER.warning(f"No faces generated for segment {seg_idx}. Skipping.")
                continue

            # 3. Save the mesh as a PLY file
            seg_mesh_output_path = output_path / f"{image_path.stem}_seg{seg_idx}_mesh.ply"
            with open(seg_mesh_output_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices_np)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")

                # Write vertex data
                for i in range(len(vertices_np)):
                    v = vertices_np[i]
                    c = vertex_colors_np[i]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

                # Write face data
                for face in faces:
                    f.write(f"3 {' '.join(map(str, face))}\n")

            LOGGER.info(f"Segment {seg_idx}: Mesh with {len(vertices_np)} vertices and {len(faces)} faces saved to {seg_mesh_output_path}")

            # 3b. Save the mesh as a GLB (glTF binary) file using trimesh
            try:
                import trimesh
                import pyvista as pv
                pv_mesh = pv.PolyData(vertices_np, np.hstack((np.full((len(faces), 1), 3, dtype=np.int64), np.array(faces, dtype=np.int64))).astype(np.int64))
                target_reduction = 0.99
                decimated = pv_mesh.decimate_pro(target_reduction, preserve_topology=True, boundary_vertex_deletion=True)

                dec_vertices = decimated.points.astype(np.float32)
                dec_faces = decimated.faces.reshape(-1, 4)[:, 1:4]

                # Map colors if possible, else use mean color
                if hasattr(decimated, 'point_arrays') and 'RGB' in decimated.point_arrays:
                    dec_colors = decimated.point_arrays['RGB'].astype(np.uint8)
                else:
                    mean_color = vertex_colors_np.mean(axis=0).astype(np.uint8)
                    dec_colors = np.tile(mean_color, (dec_vertices.shape[0], 1))

                # --- Texture and UV generation ---
                texture_path = output_path / f"{image_path.stem}_seg{seg_idx}_texture.png"
                io.save_image(color, texture_path)
                # Project each vertex to pixel coordinates using intrinsics
                uv = np.zeros((dec_vertices.shape[0], 2), dtype=np.float32)
                for i, v in enumerate(dec_vertices):
                    X, Y, Z = v
                    u_px = (f_x * X / Z) + c_x
                    v_px = (f_y * Y / Z) + c_y
                    # Normalize to [0, 1]
                    u_norm = np.clip((u_px + 0.5) / w, 0, 1)
                    v_norm = np.clip(1.0 - ((v_px + 0.5) / h), 0, 1)
                    uv[i, 0] = u_norm
                    uv[i, 1] = v_norm

                tex_image = Image.open(texture_path).convert("RGB")
                from trimesh.visual.texture import TextureVisuals
                visual = TextureVisuals(uv=uv, image=tex_image)
                dec_mesh = trimesh.Trimesh(
                    vertices=dec_vertices,
                    faces=dec_faces,
                    visual=visual,
                    process=False
                )
                seg_glb_output_path = output_path / f"{image_path.stem}_seg{seg_idx}_mesh.glb"
                dec_mesh.export(seg_glb_output_path)
                LOGGER.info(f"Segment {seg_idx}: Decimated mesh with texture saved as GLB to {seg_glb_output_path}")
            except ImportError:
                LOGGER.warning("trimesh or pyvista is not installed. GLB export skipped.")
            except Exception as e:
                LOGGER.warning(f"GLB export failed for segment {seg_idx}: {e}")

        # After processing all segments, create a combined pruned Gaussians file
        # Remove Gaussians that belong to any segment
        gaussian_means_3d = gaussians.mean_vectors[0].cpu().numpy()
        keep_gaussian = np.ones(len(gaussian_means_3d), dtype=bool)
        
        # Mark all segmented Gaussians for removal
        all_segment_indices = []
        for segment in segments:
            all_segment_indices.extend(segment)
        
        for idx in all_segment_indices:
            keep_gaussian[idx] = False

        # Create new Gaussians3D object with filtered Gaussians
        flag = torch.from_numpy(keep_gaussian).to("cpu")
        new_gaussians = Gaussians3D(
            mean_vectors=gaussians.mean_vectors[0][flag][None],
            singular_values=gaussians.singular_values[0][flag][None],
            quaternions=gaussians.quaternions[0][flag][None],
            colors=gaussians.colors[0][flag][None],
            opacities=gaussians.opacities[0][flag][None],
        )
        save_ply(new_gaussians, f_px, (height, width), mesh_pruned_output_path)
        LOGGER.info(f"Filtered Gaussians: {len(gaussian_means_3d) - len(new_gaussians.mean_vectors[0])} removed, {len(new_gaussians.mean_vectors[0])} remaining.")


def save_depth(depth_np, name):
    depth_min = np.nanmin(depth_np)
    depth_max = np.nanmax(depth_np)
    print(depth_min, depth_max)
    if depth_max > depth_min:
        depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth_np)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    if depth_gray.ndim == 3:
        depth_gray = np.squeeze(depth_gray)

    io.save_image(depth_gray, name)

def segment_gaussians_3d_bfs(
    gaussians,
    device,
    k_neighbors=10,
    normal_threshold=0.7,  # cosine similarity threshold
    distance_threshold=0.1,  # meters
    color_threshold=0.3,  # normalized color difference
):
    """
    Segment Gaussian Splats in 3D using BFS traversal.
    
    Args:
        gaussians: Gaussians3D object
        device: torch device
        k_neighbors: number of nearest neighbors to consider
        normal_threshold: cosine similarity threshold for normals (higher = more similar required)
        distance_threshold: maximum 3D distance between neighbors
        color_threshold: maximum color difference (L2 distance in [0,1] RGB space)
    
    Returns:
        List of segments, where each segment is a list of GS indices
    """
    # Get Gaussian properties
    means_3d = gaussians.mean_vectors[0].to(device)  # [N, 3]
    colors = gaussians.colors[0].to(device)  # [N, 3], assuming RGB in [0, 1]
    
    # Compute normals from covariance matrices
    # Using the smallest eigenvector as the normal direction
    quats = gaussians.quaternions[0].to(device)  # [N, 4]
    scales = gaussians.singular_values[0].to(device)  # [N, 3]
    
    # Convert quaternions to rotation matrices
    # q = [w, x, y, z]
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    
    # Rotation matrix from quaternion
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1),
    ], dim=-2)  # [N, 3, 3]
    
    # Find the axis with smallest scale (the normal direction for flat Gaussians)
    min_scale_idx = torch.argmin(scales, dim=1)  # [N]
    
    # Extract the corresponding column from rotation matrix
    normals = torch.zeros((R.shape[0], 3), device=device)
    for i in range(3):
        mask = (min_scale_idx == i)
        normals[mask] = R[mask, :, i]
    
    normals = F.normalize(normals, dim=-1)
    
    # Compute KNN using scipy KDTree (efficient, no OOM issues)
    N = means_3d.shape[0]
    LOGGER.info(f"Building KDTree for {N} Gaussians...")
    
    # Move to CPU for KDTree construction
    means_3d_cpu = means_3d.cpu().numpy()
    tree = KDTree(means_3d_cpu)
    
    # Query k+1 nearest neighbors (including self)
    LOGGER.info(f"Querying {k_neighbors} nearest neighbors...")
    knn_dists_np, knn_indices_np = tree.query(means_3d_cpu, k=k_neighbors + 1)
    
    # Exclude self (first neighbor) and convert back to torch
    knn_indices = torch.from_numpy(knn_indices_np[:, 1:]).to(device)  # [N, k]
    knn_dists = torch.from_numpy(knn_dists_np[:, 1:]).to(device).float()  # [N, k]
    
    # Get depth values (Z coordinate in camera space)
    depths = means_3d[:, 2]  # [N]
    
    # Sort points by depth (closest first)
    sorted_indices = torch.argsort(depths)
    
    N = means_3d.shape[0]
    visited = torch.zeros(N, dtype=torch.bool, device=device)
    segments = []
    
    # BFS segmentation - TEST MODE: Only process first seed
    seed_idx = sorted_indices[0].item()
    
    LOGGER.info(f"Starting BFS from seed {seed_idx} with depth {depths[seed_idx].item():.3f}")
    
    # Start a new segment
    segment = []
    queue = deque([seed_idx])
    visited[seed_idx] = True
    
    # Reference properties from seed
    seed_normal = normals[seed_idx]
    seed_color = colors[seed_idx]
    
    # Counters for debugging
    total_neighbors_checked = 0
    failed_distance = 0
    failed_normal = 0
    failed_color = 0
    
    max_segment_size = 100000  # Stop after 100k points
    
    while queue and len(segment) < max_segment_size:
        current_idx = queue.popleft()
        segment.append(current_idx)
        
        # Progress logging every 10k points
        if len(segment) % 10000 == 0:
            LOGGER.info(f"  BFS progress: {len(segment)} points, queue size: {len(queue)}")
        
        # Get current properties
        current_normal = normals[current_idx]
        current_color = colors[current_idx]
        
        # Check all neighbors
        neighbors = knn_indices[current_idx]  # [k]
        neighbor_dists = knn_dists[current_idx]  # [k]
        
        for i in range(k_neighbors):
            neighbor_idx = neighbors[i].item()
            total_neighbors_checked += 1
            
            if visited[neighbor_idx]:
                continue
            
            # Check distance threshold
            if neighbor_dists[i] > distance_threshold:
                failed_distance += 1
                continue
            
            neighbor_normal = normals[neighbor_idx]
            neighbor_color = colors[neighbor_idx]
            
            # Check normal similarity (cosine similarity with seed)
            normal_similarity = torch.dot(seed_normal, neighbor_normal)
            if normal_similarity < normal_threshold:
                failed_normal += 1
                continue
            
            # Check color similarity (L2 distance with seed)
            color_diff = torch.norm(seed_color - neighbor_color)
            if color_diff > color_threshold:
                failed_color += 1
                continue
            
            # Add to segment
            visited[neighbor_idx] = True
            queue.append(neighbor_idx)
    
    LOGGER.info(f"BFS complete: segment size = {len(segment)}")
    LOGGER.info(f"  Total neighbors checked: {total_neighbors_checked}")
    LOGGER.info(f"  Failed distance check: {failed_distance}")
    LOGGER.info(f"  Failed normal check: {failed_normal}")
    LOGGER.info(f"  Failed color check: {failed_color}")
    
    # Only keep segment if it has reasonable size
    if len(segment) > 50:  # Minimum segment size
        segments.append(segment)
        LOGGER.info(f"Segment added (size {len(segment)} > 50)")
    else:
        LOGGER.warning(f"Segment too small ({len(segment)} <= 50), not added")
    
    LOGGER.info(f"Segmented {N} Gaussians into {len(segments)} segments (TEST MODE: 1 seed only)")
    for i, seg in enumerate(segments):
        LOGGER.info(f"  Segment {i}: {len(seg)} Gaussians")
    
    return segments


if __name__ == "__main__":
    predict_cli()
