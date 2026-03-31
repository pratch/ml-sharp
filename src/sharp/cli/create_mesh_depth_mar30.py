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

    if laplacian_mask is None:
        lap = np.zeros((rows, cols), dtype=np.uint8)
    else:
        lap = (np.asarray(laplacian_mask) != 0).astype(np.uint8)

    dx = (1, -1, 0, 0)
    dy = (0, 0, 1, -1)

    for y in range(rows):
        for x in range(cols):
            if lap[y, x] != 0:
                mask[y, x] = 255
                continue

            d = float(depth[y, x])
            if d == 0.0:
                continue

            is_edge = False

            # for each of 4 neighbors, if depth diff >= threshold, mark as edge
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if 0 <= nx < cols and 0 <= ny < rows:
                    dn = float(depth[ny, nx])
                    if not pass_threshold(d, dn, threshold, exponent):
                        is_edge = True
                        break

            if is_edge:
                mask[y, x] = 255

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

    vertices: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    faces: list[list[int]] = []
    vertex_idx = np.full((h, w), -1, dtype=np.int32)

    for y in range(h):
        for x in range(w):
            d = float(depth[y, x])
            if d <= 0.0:
                continue

            x_cam = (x - cx) * d / fx
            y_cam = (y - cy) * d / fy
            z_cam = d

            pt_world = ext_inv @ np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
            vertex_idx[y, x] = len(vertices)
            vertices.append(pt_world[:3].astype(np.float32))

            c = img_np[y, x]
            colors.append(np.array([c[0], c[1], c[2]], dtype=np.float32) / 255.0)

            u = (fx * (x_cam / z_cam) + cx) / float(w)
            v = (fy * (y_cam / z_cam) + cy) / float(h)
            uvs.append(np.array([u, v], dtype=np.float32))

    coords = ((0, 0), (1, 0), (0, 1), (1, 1))
    tset = ((0, 2, 1), (3, 1, 2))

    if external_depth is None:
        ext_depth = None
    else:
        ext_depth = np.squeeze(np.asarray(external_depth, dtype=np.float32))

    for y in range(h - 1):
        for x in range(w - 1):
            for tri in tset:
                v_indices = [0, 0, 0]
                skip = False

                for j in range(3):
                    vx = x + coords[tri[j]][0]
                    vy = y + coords[tri[j]][1]
                    vid = int(vertex_idx[vy, vx])
                    v_indices[j] = vid

                    if vid == -1 or edge_mask[vy, vx] != 0:
                        skip = True
                        break

                    if ext_depth is not None:
                        d_ext = float(ext_depth[vy, vx])
                        if d_ext > 1e-6 and float(depth[vy, vx]) > d_ext - 0.01 * d_ext * d_ext:
                            skip = True
                            break

                    nx = x + coords[tri[(j + 1) % 3]][0]
                    ny = y + coords[tri[(j + 1) % 3]][1]
                    if lap_mask[vy, vx] != 0 or lap_mask[ny, nx] != 0:
                        skip = True
                        break

                    if not pass_threshold(float(depth[vy, vx]), float(depth[ny, nx]), threshold, exponent):
                        skip = True
                        break

                if not skip:
                    faces.append(v_indices)

    vertices_np = np.asarray(vertices, dtype=np.float32) if vertices else np.zeros((0, 3), dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    colors_np = np.asarray(colors, dtype=np.float32) if colors else np.zeros((0, 3), dtype=np.float32)
    uvs_np = np.asarray(uvs, dtype=np.float32) if uvs else np.zeros((0, 2), dtype=np.float32)

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
def predict_cli(
    plyfile: Path,
    output_path: Path,
    device: str,
    verbose: bool,
    radius: float,
    camera_mode: str,
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

    stem = ply_path.stem
    io.save_image(color, output_path / f"{stem}_rendered.png")

    colored_depth_pt = vis.colorize_depth(depth, min(depth_np.max(), vis.METRIC_DEPTH_MAX_CLAMP_METER))
    colored_depth_np = colored_depth_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    io.save_image(colored_depth_np, output_path / f"{stem}_rawdepth.png")

    # build mesh and save to .PLY
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

    mesh_path = output_path / f"{stem}_mesh.ply"
    write_ply_file(mesh_path, mesh.vertices, mesh.faces, mesh.colors)

    LOGGER.info(
        "Saved extracted mesh for %s (depth min=%.4f, max=%.4f, verts=%d, faces=%d)",
        stem,
        float(np.nanmin(depth_np)),
        float(np.nanmax(depth_np)),
        int(mesh.vertices.shape[0]),
        int(mesh.faces.shape[0]),
    )
    LOGGER.info("Saved mesh PLY to %s", mesh_path)


if __name__ == "__main__":
    predict_cli()
