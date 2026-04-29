# Rule: tmux send -t 1 '!dpython' Enter Enter
"""Render Gaussians and depth maps from multiple cameras.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import click
import cv2
import numpy as np
import torch
import tqdm
import time

try:
    import xatlas
except ImportError:
    xatlas = None

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras
except ImportError:
    Meshes = None
import xatlas
from PIL import Image

# PyTorch3D imports
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    PerspectiveCameras
)

from sharp.utils import io, gsplat
from sharp.utils import vis
from sharp.utils.gaussians import load_ply

from sharp.cli.camera_utils import compute_w2c, place_camera_spherical   
from sharp.cli.mesh_utils import write_ply_file

LOGGER = logging.getLogger(__name__)


@dataclass
class MeshResult:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    texture: np.ndarray
    uvs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))


def pass_threshold(d1: float, d2: float, threshold: float, exponent: float = -1.0) -> bool:
    if d1 <= 0.0 or d2 <= 0.0:
        return False

    if exponent == 0.0:
        return abs(np.log(d1) - np.log(d2)) < threshold
    if exponent == 1.0:
        return abs(d1 - d2) < threshold
    if exponent == -1.0:
        return abs(d1 - d2) < threshold * d1 * d1

    v1 = np.power(d1, exponent)
    v2 = np.power(d2, exponent)
    return abs(v1 - v2) < threshold


def find_edge(
    depth_map: np.ndarray,
    img: np.ndarray,
    threshold: float,
    exponent: float = -1.0,
    laplacian_mask: np.ndarray | None = None,
) -> np.ndarray:
    depth = np.squeeze(np.asarray(depth_map, dtype=np.float32))
    rows, cols = depth.shape[:2]
    mask = np.zeros((rows, cols), dtype=np.uint8)

    if laplacian_mask is not None:
        lap = (np.asarray(laplacian_mask) != 0)
        mask[lap] = 255
    else:
        lap = np.zeros((rows, cols), dtype=bool)

    d = depth
    valid = d > 0.0

    def check_edge(dn):
        both_valid = valid & (dn > 0.0)
        if exponent == 0.0:
            diff = np.abs(np.log(np.maximum(d, 1e-8)) - np.log(np.maximum(dn, 1e-8)))
            is_edge = diff >= threshold
        elif exponent == 1.0:
            diff = np.abs(d - dn)
            is_edge = diff >= threshold
        elif exponent == -1.0:
            diff = np.abs(d - dn)
            is_edge = diff >= threshold * d * d
        else:
            v1 = np.power(d, exponent)
            v2 = np.power(dn, exponent)
            diff = np.abs(v1 - v2)
            is_edge = diff >= threshold
        return both_valid & is_edge

    dn_right = np.empty_like(d)
    dn_right[:, :-1] = d[:, 1:]
    dn_right[:, -1] = 0.0
    edge_mask = check_edge(dn_right)

    dn_left = np.empty_like(d)
    dn_left[:, 1:] = d[:, :-1]
    dn_left[:, 0] = 0.0
    edge_mask |= check_edge(dn_left)

    dn_down = np.empty_like(d)
    dn_down[:-1, :] = d[1:, :]
    dn_down[-1, :] = 0.0
    edge_mask |= check_edge(dn_down)

    dn_up = np.empty_like(d)
    dn_up[1:, :] = d[:-1, :]
    dn_up[0, :] = 0.0
    edge_mask |= check_edge(dn_up)

    edge_mask &= ~lap
    mask[edge_mask] = 255
    return mask


def _remove_small_components_common(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    uvs: np.ndarray,
    min_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """remove small islands of connected components of faces"""
    if faces.size == 0 or min_faces <= 0:
        return vertices, faces, colors, uvs

    num_faces = faces.shape[0]
    parent = np.arange(num_faces, dtype=np.int32)
    comp_size = np.ones(num_faces, dtype=np.int32)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = int(parent[root])
        while parent[i] != root:
            nxt = int(parent[i])
            parent[i] = root
            i = nxt
        return root

    def unite(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri == rj:
            return
        if comp_size[ri] < comp_size[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        comp_size[ri] += comp_size[rj]

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, f in enumerate(faces):
        e01 = tuple(sorted((int(f[0]), int(f[1]))))
        e12 = tuple(sorted((int(f[1]), int(f[2]))))
        e20 = tuple(sorted((int(f[2]), int(f[0]))))
        edge_to_faces.setdefault(e01, []).append(fi)
        edge_to_faces.setdefault(e12, []).append(fi)
        edge_to_faces.setdefault(e20, []).append(fi)

    for flist in edge_to_faces.values():
        if len(flist) > 1:
            base = flist[0]
            for other in flist[1:]:
                unite(base, other)

    keep_face = np.zeros(num_faces, dtype=bool)
    for i in range(num_faces):
        keep_face[i] = comp_size[find(i)] >= min_faces

    new_faces = faces[keep_face]
    if new_faces.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    used_vertices = np.unique(new_faces.reshape(-1))
    remap = {int(v): i for i, v in enumerate(used_vertices.tolist())}

    remapped_faces = np.array(
        [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in new_faces],
        dtype=np.int32,
    )

    return (
        vertices[used_vertices],
        remapped_faces,
        colors[used_vertices] if colors.size else colors,
        uvs[used_vertices] if uvs.size else uvs,
    )


def create_mesh(
    depth_map: np.ndarray,
    img: np.ndarray,
    K: np.ndarray,
    ext: np.ndarray,
    threshold: float = 0.01,
    min_component_size: int = 100,
    dilate_iterations: int = 1,
    external_depth: np.ndarray | None = None,
    exponent: float = -1.0,
    laplacian_mask: np.ndarray | None = None,
) -> tuple[MeshResult, np.ndarray]:
    """create mesh based on GS depth"""
    depth = np.squeeze(depth_map.detach().cpu().numpy().astype(np.float32, copy=False))

    K_np = K.detach().cpu().numpy().astype(np.float32, copy=False)
    ext_np = ext.detach().cpu().numpy().astype(np.float32, copy=False)
    ext_inv = np.linalg.inv(ext_np)

    h, w = depth.shape[:2]
    img_np = np.squeeze(np.asarray(img))
    if img_np.dtype != np.float32:
        img_np = img_np.astype(np.float32)

    if laplacian_mask is None:
        lap_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        lap_mask = (np.asarray(laplacian_mask) != 0).astype(np.uint8)

    # find edges where depth discont is high, so we don't mesh there
    edge_mask = find_edge(depth, img_np, threshold, exponent, lap_mask)
    if dilate_iterations > 0:
        edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), dtype=np.uint8), iterations=dilate_iterations)

    fx = float(K_np[0, 0])
    fy = float(K_np[1, 1])
    cx = float(K_np[0, 2])
    cy = float(K_np[1, 2])

    valid = depth > 0.0
    y_coords, x_coords = np.nonzero(valid)
    num_valid = len(y_coords)

    vertex_idx = np.full((h, w), -1, dtype=np.int32)
    vertex_idx[valid] = np.arange(num_valid, dtype=np.int32)

    d_valid = depth[valid]
    x_cam = (x_coords - cx) * d_valid / fx
    y_cam = (y_coords - cy) * d_valid / fy
    z_cam = d_valid

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=0)
    pts_world = (ext_inv @ pts_cam)[:3, :].T.astype(np.float32)
    vertices_np = pts_world

    colors_np = (img_np[valid] / 255.0).astype(np.float32)

    u_coords = (fx * (x_cam / z_cam) + cx) / float(w)
    v_coords = (fy * (y_cam / z_cam) + cy) / float(h)
    uvs_np = np.stack([u_coords, v_coords], axis=1).astype(np.float32)

    if external_depth is None:
        ext_depth = None
    else:
        ext_depth = np.squeeze(np.asarray(external_depth, dtype=np.float32))

    invalid_mask = (vertex_idx == -1) | (edge_mask != 0) | (lap_mask != 0)
    
    if ext_depth is not None:
        invalid_ext = (ext_depth > 1e-6) & (depth > ext_depth - 0.01 * ext_depth * ext_depth)
        invalid_mask |= invalid_ext

    def check_invalid_edge_vec(d1, d2):
        valid_both = (d1 > 0.0) & (d2 > 0.0)
        if exponent == 0.0:
            diff = np.abs(np.log(np.maximum(d1, 1e-8)) - np.log(np.maximum(d2, 1e-8)))
        elif exponent == -1.0:
            diff = np.abs(d1 - d2)
            return valid_both & (diff >= threshold * d1 * d1)
        else:
            diff = np.abs(d1 - d2)
        return valid_both & (diff >= threshold)

    v00 = vertex_idx[:-1, :-1]
    v10 = vertex_idx[:-1, 1:]
    v01 = vertex_idx[1:, :-1]
    v11 = vertex_idx[1:, 1:]
    
    inv00 = invalid_mask[:-1, :-1]
    inv10 = invalid_mask[:-1, 1:]
    inv01 = invalid_mask[1:, :-1]
    inv11 = invalid_mask[1:, 1:]
    
    d00 = depth[:-1, :-1]
    d10 = depth[:-1, 1:]
    d01 = depth[1:, :-1]
    d11 = depth[1:, 1:]
    
    edge00_01 = check_invalid_edge_vec(d00, d01)
    edge01_10 = check_invalid_edge_vec(d01, d10)
    edge10_00 = check_invalid_edge_vec(d10, d00)
    
    edge11_10 = check_invalid_edge_vec(d11, d10)
    edge10_01 = check_invalid_edge_vec(d10, d01)
    edge01_11 = check_invalid_edge_vec(d01, d11)
    
    valid_t1 = ~(inv00 | inv01 | inv10 | edge00_01 | edge01_10 | edge10_00)
    valid_t2 = ~(inv11 | inv10 | inv01 | edge11_10 | edge10_01 | edge01_11)
    
    f_t1 = np.stack([v00[valid_t1], v01[valid_t1], v10[valid_t1]], axis=1)
    f_t2 = np.stack([v11[valid_t2], v10[valid_t2], v01[valid_t2]], axis=1)
    
    if len(f_t1) > 0 and len(f_t2) > 0:
        faces_np = np.concatenate([f_t1, f_t2], axis=0).astype(np.int32)
    elif len(f_t1) > 0:
        faces_np = f_t1.astype(np.int32)
    elif len(f_t2) > 0:
        faces_np = f_t2.astype(np.int32)
    else:
        faces_np = np.zeros((0, 3), dtype=np.int32)

    if min_component_size > 0:
        vertices_np, faces_np, colors_np, uvs_np = _remove_small_components_common(
            vertices_np,
            faces_np,
            colors_np,
            uvs_np,
            min_component_size,
        )

    # create texture image
    texture = np.clip(img_np, 0, 255).astype(np.uint8)
    mesh = MeshResult(vertices=vertices_np, faces=faces_np, colors=colors_np, texture=texture, uvs=uvs_np)
    return mesh, depth


def get_pytorch3d_cameras(ext, K, hw, device):
    """Convert OpenCV extrinsics to PyTorch3D PerspectiveCameras."""
    R = ext[:3, :3]
    T = ext[:3, 3]

    # OpenCV to PyTorch3D coordinate system transformation
    # OpenCV: X right, Y down, Z forward
    # PyTorch3D: X left, Y up, Z forward (in NDC) / object space
    R_pt3d = R.clone().T
    R_pt3d[:, 0] *= -1.0 # Invert X
    R_pt3d[:, 1] *= -1.0 # Invert Y
    
    T_pt3d = T.clone()
    T_pt3d[0] *= -1.0
    T_pt3d[1] *= -1.0

    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]

    focal_length = torch.stack([fx, fy]).unsqueeze(0)
    principal_point = torch.stack([px, py]).unsqueeze(0)

    cameras = PerspectiveCameras(
        device=device,
        R=R_pt3d.unsqueeze(0),
        T=T_pt3d.unsqueeze(0),
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        image_size=(hw,)
    )
    return cameras


def prune_faces_by_carving(mesh, gaussians, metadata, renderer, K, ext_base, num_views, radius, guilt_ratio, device):
    """Multi-view carving: render GS and Mesh depths, prune faces with high guilt."""
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    
    if len(faces) == 0:
        return faces

    num_faces = faces.shape[0]
    guilt = torch.zeros(num_faces, dtype=torch.float32, device=device)
    observed = torch.zeros(num_faces, dtype=torch.float32, device=device)
    
    h, w = metadata.resolution_px[1], metadata.resolution_px[0]
    hw = (h, w)

    gs_means = gaussians.mean_vectors[0].detach().cpu().numpy()
    mean_pos = np.mean(gs_means, axis=0)
    
    import tqdm
    
    meshes = Meshes(verts=[vertices], faces=[faces])
    
    for i in tqdm.tqdm(range(num_views), desc="Carving faces"):
        az = (i / num_views) * 360.0
        cam_pos_np = place_camera_spherical(mean_pos, radius, az=az, el=0.0)
        ext_view = compute_w2c(camera_positions=cam_pos_np, target_positions=mean_pos)
        ext_view = ext_view.to(device)

        # 1. Render GS depth
        gs_out = renderer(
            gaussians.to(device),
            extrinsics=ext_view.unsqueeze(0),
            intrinsics=K.unsqueeze(0),
            image_width=w,
            image_height=h,
        )
        d_gs = gs_out.depth[0].squeeze(0) # (H, W)

        # 2. Render Mesh depth
        cameras = get_pytorch3d_cameras(ext_view, K, hw, device)
        hw_int = (int(hw[0]), int(hw[1])) if isinstance(hw, tuple) else (int(hw), int(hw))
        raster_settings = RasterizationSettings(
            image_size=hw_int, 
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
            max_faces_per_bin=100000,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(meshes)
        
        d_mesh = fragments.zbuf.squeeze(-1)[0] # (H, W)
        pix_to_face = fragments.pix_to_face.squeeze(-1)[0] # (H, W)
        
        # 3. Accumulate guilt
        valid_mask = (pix_to_face >= 0) & (d_gs > 0) & (d_mesh > 0)
        if not valid_mask.any():
            continue
            
        d_gs_valid = d_gs[valid_mask]
        d_mesh_valid = d_mesh[valid_mask]
        faces_valid = pix_to_face[valid_mask]
        
        # Condition: GS is significantly behind Mesh -> space is empty but mesh is there!
        guilty_pixels = d_gs_valid > (d_mesh_valid + 0.01 * d_mesh_valid**2)
        
        # Scatter add
        ones = torch.ones_like(faces_valid, dtype=torch.float32)
        observed.scatter_add_(0, faces_valid.long(), ones)
        guilt.scatter_add_(0, faces_valid.long(), guilty_pixels.float())

    # 4. Prune faces
    ratio = guilt / (observed + 1e-6)
    keep_mask = ratio <= guilt_ratio
    pruned_faces = faces[keep_mask]
    
    return pruned_faces.cpu().numpy()


def prune_splats_by_visibility(gaussians, metadata, mesh, ext_base, K, num_views, radius, device):
    """Mark splats visible from exterior cameras. Slices Gaussians3D."""
    gs_means = gaussians.mean_vectors[0].to(device)
    num_splats = gs_means.shape[0]
    visible_mask = torch.zeros(num_splats, dtype=torch.bool, device=device)
    
    h, w = metadata.resolution_px[1], metadata.resolution_px[0]
    hw = (h, w)
    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    meshes = Meshes(verts=[vertices], faces=[faces]) if len(faces) > 0 else None
    
    mean_pos_np = gs_means.detach().cpu().numpy().mean(axis=0)
    
    for i in tqdm.tqdm(range(num_views), desc="Pruning splats"):
        az = (i / num_views) * 360.0
        cam_pos_np = place_camera_spherical(mean_pos_np, radius, az=az, el=0.0)
        ext_view = compute_w2c(camera_positions=cam_pos_np, target_positions=mean_pos_np)
        ext_view = ext_view.to(device)

        # 1. Render Mesh depth
        if meshes is not None:
            cameras = get_pytorch3d_cameras(ext_view, K, hw, device)
            hw_int = (int(hw[0]), int(hw[1])) if isinstance(hw, tuple) else (int(hw), int(hw))
            raster_settings = RasterizationSettings(
                image_size=hw_int, 
                blur_radius=0.0,
                faces_per_pixel=1,
                max_faces_per_bin=100000,
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(meshes)
            d_mesh = fragments.zbuf.squeeze(-1)[0] # (H, W)
        else:
            d_mesh = torch.zeros((h, w), dtype=torch.float32, device=device) + 1e5

        # 2. Project splats
        R = ext_view[:3, :3]
        T = ext_view[:3, 3]
        pts_cam = (R @ gs_means.T).T + T
        
        fx, fy = K[0, 0], K[1, 1]
        px, py = K[0, 2], K[1, 2]
        
        z_cam = pts_cam[:, 2]
        x_cam = pts_cam[:, 0]
        y_cam = pts_cam[:, 1]
        
        # Valid depths
        z_mask = z_cam > 0
        
        # Project
        u = (fx * x_cam / z_cam + px).long()
        v = (fy * y_cam / z_cam + py).long()
        
        uv_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        valid = z_mask & uv_mask
        
        u_valid = u[valid]
        v_valid = v[valid]
        z_splat_valid = z_cam[valid]
        
        d_mesh_at_splats = d_mesh[v_valid, u_valid]
        
        # Splats in front of mesh or not occluded by mesh
        # Provide small margin
        visible = z_splat_valid <= (d_mesh_at_splats + 0.05)
        
        # Update global mask
        valid_indices = torch.nonzero(valid).squeeze(-1)
        visible_mask[valid_indices[visible]] = True

    # Slice gaussians
    # Gaussians3D is a NamedTuple, so we must slice its underlying tensors along the N dimension
    # Tensors are typically shape (1, N, D), so we slice dim 1
    mask = visible_mask.to(gaussians.mean_vectors.device)
    
    from sharp.utils.gaussians import Gaussians3D
    sliced_gaussians = Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask],
        singular_values=gaussians.singular_values[:, mask],
        quaternions=gaussians.quaternions[:, mask],
        colors=gaussians.colors[:, mask],
        opacities=gaussians.opacities[:, mask],
    )
    return sliced_gaussians


def generate_atlas_xatlas(vertices, faces, name="atlas"):
    v_atlas, ids, uvs = xatlas.parametrize(vertices.cpu().numpy(), faces.cpu().numpy())
    return v_atlas, ids, uvs


def simplify_mesh_25d_open3d(vertices, faces, K, ext, target_faces, exponent=-1.0):
    try:
        import open3d as o3d
    except ImportError:
        LOGGER.warning("Open3D not found. Skipping 2.5D simplification.")
        return vertices, faces
        
    N = vertices.shape[0]
    pts = np.hstack([vertices, np.ones((N, 1))])
    pts_cam = (ext @ pts.T).T[:, :3]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid_z = np.maximum(z, 1e-6)
    
    u = fx * x / valid_z + cx
    v = fy * y / valid_z + cy
    
    if exponent == 0.0:
        z_t = np.log(valid_z)
    else:
        z_t = np.power(valid_z, exponent)
        
    uvz = np.stack([u, v, z_t], axis=-1)
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(uvz)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    LOGGER.info(f"Simplifying 2.5D mesh from {len(faces)} to {target_faces} faces...")
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    
    uvz_s = np.asarray(o3d_mesh.vertices)
    faces_s = np.asarray(o3d_mesh.triangles)
    
    u_s = uvz_s[:, 0]
    v_s = uvz_s[:, 1]
    z_t_s = uvz_s[:, 2]
    
    if exponent == 0.0:
        z_s = np.exp(z_t_s)
    else:
        z_s = np.power(np.maximum(z_t_s, 1e-6), 1.0 / exponent)
        
    x_s = (u_s - cx) * z_s / fx
    y_s = (v_s - cy) * z_s / fy
    
    pts_cam_s = np.stack([x_s, y_s, z_s, np.ones_like(z_s)], axis=-1)
    c2w = np.linalg.inv(ext)
    pts_world_s = (c2w @ pts_cam_s.T).T[:, :3]
    
    LOGGER.info(f"Simplification done. New vertices: {len(pts_world_s)}, New faces: {len(faces_s)}")
    return pts_world_s.astype(np.float32), faces_s.astype(np.int32)


@click.command()
@click.option(
    "--plyfile",
    type=click.Path(path_type=Path),
    default="lego.ply",
    show_default=True,
    help="PLY filename or path to render.",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save rendered images/depth maps.",
    required=True,
)
@click.option("--device", type=str, default="cuda", help="Device to run on.")
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
@click.option("--radius", type=float, default=2.5, help="Camera distance from object center in meters.")
@click.option(
    "--camera-mode",
    type=click.Choice(["identity", "spherical", "ply", "auto"], case_sensitive=False),
    default="spherical",
    show_default=True,
    help="How to choose extrinsics for rendering/meshing.",
)
@click.option("--num-circle-views", type=int, default=36, help="Views for carving/atlas")
@click.option("--carving-radius", type=float, default=2.5, help="Radius for carving cameras")
@click.option("--guilt-ratio", type=float, default=0.2, help="Ratio for carving splat rejection")
@click.option("--simplify-factor", type=float, default=0.05, help="Fraction of faces to keep (2.5D simplification). 1.0 or 0.0 means skip.")
def predict_cli(
    plyfile: Path,
    output_path: Path,
    device: str,
    verbose: bool,
    radius: float,
    camera_mode: str,
    num_circle_views: int,
    carving_radius: float,
    guilt_ratio: float,
    simplify_factor: float,
) -> None:

    output_path.mkdir(parents=True, exist_ok=True)
    
    # set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    log_path = output_path / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)

    LOGGER.info("Output directory: %s", output_path)
    LOGGER.info("Logging to file: %s", log_file)

    device = torch.device(device if device != "default" else "cuda")
    
    ply_path = plyfile
    if not ply_path.exists():
        ply_path = Path("splats") / plyfile.name
    if not ply_path.exists():
        raise click.ClickException(f"PLY file not found: {plyfile}")

    gaussians, metadata, _, input_extrinsics = load_ply(ply_path)
    width, height = metadata.resolution_px
    f_px = metadata.focal_length_px

    # print("Color space:", metadata.color_space)
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)

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

    camera_mode = camera_mode.lower()
    if camera_mode == "identity":
        extrinsics = torch.eye(4, dtype=torch.float32)
        resolved_camera_mode = "identity"
    elif camera_mode == "ply":
        extrinsics = input_extrinsics.float()
        resolved_camera_mode = "ply"
    elif camera_mode == "spherical":
        gs_means = gaussians.mean_vectors[0].cpu().numpy()
        mean_pos = np.mean(gs_means, axis=0)
        cam_pos = place_camera_spherical(mean_pos, radius, az=-90.0, el=0.0)
        extrinsics = compute_w2c(camera_positions=cam_pos, target_positions=mean_pos)
        resolved_camera_mode = "spherical"
    else:
        eye = torch.eye(4, dtype=input_extrinsics.dtype, device=input_extrinsics.device)
        if torch.allclose(input_extrinsics, eye, atol=1e-4):
            extrinsics = torch.eye(4, dtype=torch.float32)
            resolved_camera_mode = "auto->identity"
        else:
            extrinsics = input_extrinsics.float()
            resolved_camera_mode = "auto->ply"

    LOGGER.info("Camera mode: %s", resolved_camera_mode)

    # render GS and depth
    render_start = time.time()
    rendering_output = renderer(
        gaussians.to(device),
        extrinsics=extrinsics[None].to(device),
        intrinsics=intrinsics[None],
        image_width=width,
        image_height=height,
    )

    depth = rendering_output.depth[0]
    depth_np = depth.cpu().numpy()
    color = torch.clamp(rendering_output.color[0], 0.0, 1.0)
    color = (color.permute(1, 2, 0) * 255.0).to(dtype=torch.uint8).cpu().numpy()
    LOGGER.info(f"GS depth rendering completed in {time.time() - render_start:.2f}s")

    stem = ply_path.stem
    io.save_image(color, output_path / f"{stem}_rendered.png")

    colored_depth_pt = vis.colorize_depth(depth, min(depth_np.max(), vis.METRIC_DEPTH_MAX_CLAMP_METER))
    colored_depth_np = colored_depth_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    io.save_image(colored_depth_np, output_path / f"{stem}_rawdepth.png")

    # build mesh and save to .PLY
    mesh_start = time.time()
    mesh, _ = create_mesh(
        depth_map=depth,
        img=color,
        K=intrinsics,
        ext=extrinsics,
        threshold=0.01,
        min_component_size=100,
        dilate_iterations=1,
        external_depth=None,
        exponent=-1.0,
        laplacian_mask=None,
    )
    LOGGER.info(f"Initial mesh extraction completed in {time.time() - mesh_start:.2f}s")
    
    # Save the initial dense mesh before simplification
    dense_mesh_path = output_path / f"{stem}_mesh_initial.ply"
    write_ply_file(dense_mesh_path, mesh.vertices, mesh.faces, mesh.colors)
    LOGGER.info(
        "Saved initial dense mesh for %s (depth min=%.4f, max=%.4f, verts=%d, faces=%d)",
        stem,
        float(np.nanmin(depth_np)),
        float(np.nanmax(depth_np)),
        int(mesh.vertices.shape[0]),
        int(mesh.faces.shape[0]),
    )

    if 0.0 < simplify_factor < 1.0:
        simp_start = time.time()
        target_faces = int(len(mesh.faces) * simplify_factor)
        LOGGER.info(f"Targeting {target_faces} faces for 2.5D simplification (factor {simplify_factor}).")
        simp_verts, simp_faces = simplify_mesh_25d_open3d(
            mesh.vertices,
            mesh.faces,
            intrinsics.cpu().numpy(),
            extrinsics.cpu().numpy(),
            target_faces,
            exponent=-1.0
        )
        mesh.vertices = simp_verts
        mesh.faces = simp_faces
        # colors won't perfectly map without KDTree interpolation, but we skip that for now
        # since atlas only needs plain geometry
        mesh.colors = np.ones((len(simp_verts), 3), dtype=np.uint8) * 200
        LOGGER.info(f"2.5D Simplification completed in {time.time() - simp_start:.2f}s")

    mesh_path = output_path / f"{stem}_mesh_simplified.ply"
    write_ply_file(mesh_path, mesh.vertices, mesh.faces, mesh.colors)
    LOGGER.info(f"Saved simplified mesh PLY to {mesh_path} (verts={int(mesh.vertices.shape[0])}, faces={int(mesh.faces.shape[0])})")

    # Ported from C++: carving, atlas, pruning visible splats
    LOGGER.info("Starting multi-view carving and atlas generation...")
    
    # 3. Multi-view carving
    carve_start = time.time()
    mesh.faces = prune_faces_by_carving(
        mesh=mesh,
        gaussians=gaussians,
        metadata=metadata,
        renderer=renderer,
        K=intrinsics,
        ext_base=extrinsics,
        num_views=num_circle_views,
        radius=carving_radius,
        guilt_ratio=guilt_ratio,
        device=device,
    )
    LOGGER.info(f"Multi-view carving completed in {time.time() - carve_start:.2f}s")
    LOGGER.info(f"Pruned mesh faces to {len(mesh.faces)}")

    # 4. visibility
    LOGGER.info("Pruning splats by visibility...")
    vis_start = time.time()
    gaussians_visible = prune_splats_by_visibility(
        gaussians=gaussians,
        metadata=metadata,
        mesh=mesh,
        ext_base=extrinsics,
        K=intrinsics,
        num_views=num_circle_views,
        radius=carving_radius,
        device=device,
    )
    LOGGER.info(f"Splat visibility pruning completed in {time.time() - vis_start:.2f}s")
    
    # 5. Atlas generation wrapping xatlas
    # if xatlas is not None:
    #     LOGGER.info("Generating texture atlas...")
    #     atlas_start = time.time()
    #     verts_tensor = torch.tensor(mesh.vertices, device=device)
    #     faces_tensor = torch.tensor(mesh.faces, device=device)
    #     v_atlas, ids, uvs = generate_atlas_xatlas(verts_tensor, faces_tensor)
    #     mesh.uvs = uvs
    #     LOGGER.info(f"Texture atlas generated in {time.time() - atlas_start:.2f}s")
    # else:
    #     LOGGER.info("xatlas not found, skipping texture atlas.")
    
    # Save the updated mesh
    pruned_mesh_path = output_path / f"{stem}_mesh_pruned.ply"
    write_ply_file(pruned_mesh_path, mesh.vertices, mesh.faces, mesh.colors)
    LOGGER.info(f"Saved pruned mesh PLY to {pruned_mesh_path}")

    # 6. Save visible_splats.ply
    save_splat_start = time.time()
    visible_splats_path = output_path / f"{stem}_visible_splats.ply"
    LOGGER.info(f"Saving {len(gaussians_visible.mean_vectors[0])} pruned visible splats to {visible_splats_path}")
    from sharp.utils.gaussians import save_ply
    save_ply(gaussians_visible, metadata.focal_length_px, metadata.resolution_px, visible_splats_path)
    LOGGER.info(f"Saved visible splats in {time.time() - save_splat_start:.2f}s")
    LOGGER.info("Pipeline completed successfully.")

if __name__ == "__main__":
    predict_cli()
