import torch
import numpy as np

def place_camera_spherical(center: np.ndarray, radius: float, az: float, el: float) -> np.ndarray:
    """Place camera at given spherical coordinates around center."""
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    cam_pos = center + radius * np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.sin(el_rad),
        np.cos(el_rad) * np.sin(az_rad),
    ], dtype=np.float32)
    return cam_pos


def compute_w2c(
    camera_positions: np.ndarray,
    target_positions: np.ndarray,
    up: np.ndarray | None = None,
) -> torch.Tensor:
    """Compute world-to-camera extrinsics in OpenCV convention."""
    cam_pos = np.atleast_2d(np.asarray(camera_positions, dtype=np.float32))
    target_pos = np.atleast_2d(np.asarray(target_positions, dtype=np.float32))

    if up is None:
        up_arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    else:
        up_arr = np.atleast_2d(np.asarray(up, dtype=np.float32))

    batch_size = max(len(cam_pos), len(target_pos), len(up_arr))
    if len(cam_pos) == 1:
        cam_pos = np.repeat(cam_pos, batch_size, axis=0)
    if len(target_pos) == 1:
        target_pos = np.repeat(target_pos, batch_size, axis=0)
    if len(up_arr) == 1:
        up_arr = np.repeat(up_arr, batch_size, axis=0)

    extrinsics_list = []
    for i in range(batch_size):
        eye = cam_pos[i]
        target = target_pos[i]
        up_vec = up_arr[i]

        z_axis = target - eye
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

        x_axis = np.cross(up_vec, z_axis)
        x_axis_norm = np.linalg.norm(x_axis)
        if x_axis_norm < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
            x_axis = np.cross(x_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        t = -R @ eye

        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = R
        ext[:3, 3] = t
        extrinsics_list.append(ext)

    extrinsics = np.stack(extrinsics_list, axis=0)
    if np.asarray(camera_positions).ndim == 1 and np.asarray(target_positions).ndim == 1:
        return torch.from_numpy(extrinsics[0])
    return torch.from_numpy(extrinsics)