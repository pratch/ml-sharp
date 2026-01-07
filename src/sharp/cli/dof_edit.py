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

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    load_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


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

def dof_edit_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)


    output_path.mkdir(exist_ok=True, parents=True)

    input_ply = output_path / "20260104_171141_original.ply"

    

    gaussians, metadata, _, _ = load_ply(input_ply)

    depths = gaussians.mean_vectors[..., 2].cpu().numpy().flatten()
    median_depth = float(np.median(depths))

    # loop over all splats
    print(gaussians.mean_vectors.shape)
    print(gaussians.singular_values.shape)
    # for i in range(gaussians.mean_vectors.shape[1]):
    #     depth = gaussians.mean_vectors[0, i, 2].item()
    #     depth_offset = depth - median_depth
    #
    #     scale = 1.0 + 10 * (depth_offset / 2.0) ** 2
    #     gaussians.singular_values[0, i] *= scale
    #     gaussians.mean_vectors[0, i, 0] + 5
    median_depth = float(np.median(depths)) * 0.01
    depth_offsets = depths - median_depth
    scales = 1.0 + 20 * (depth_offsets / 2.0) ** 2
    gaussians.singular_values[0] *= torch.from_numpy(scales).to(gaussians.singular_values.device).view(-1, 1)
    # Example vectorized operation for mean_vectors (if you want to add 5 to x)
    # gaussians.mean_vectors[0, :, 0] += 5

    image, _, f_px = io.load_rgb(input_path / "20260104_171141.jpg")
    height, width = image.shape[:2]

    save_ply(gaussians, f_px, (height, width), output_path / "20260104_171141.ply")
