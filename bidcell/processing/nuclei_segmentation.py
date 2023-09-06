import argparse
import os

import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from skimage.transform import resize
from tqdm import tqdm

from .utils import get_patches_coords
from ..config import Config, load_config


def resize_dapi(dapi, new_h, new_w):
    """Resize DAPI image"""
    resized = resize(dapi, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    return resized


def segment_dapi(img, diameter=None, use_cpu=False):
    """Segment nuclei in DAPI image using Cellpose"""
    use_gpu = True if not use_cpu else False
    model = models.Cellpose(gpu=use_gpu, model_type="cyto")
    channels = [0, 0]
    mask, _, _, _ = model.eval(img, diameter=diameter, channels=channels)
    return mask


def segment_nuclei(config: Config):
    dir_dataset = config.files.data_dir

    print("Reading DAPI image")
    if config.files.fp_dapi is None:
        fp_dapi = os.path.join(dir_dataset, "dapi_stitched.tif")
    else:
        fp_dapi = config.files.fp_dapi
    print(fp_dapi)
    dapi = tifffile.imread(fp_dapi)

    # Crop to size of transcript map (requires getting transcript maps first)
    if config.nuclei.crop_nuclei_to_ts:
        # Get starting coordinates
        fp_affine = os.path.join(dir_dataset, config.files.fp_affine)

        affine = pd.read_csv(fp_affine, index_col=0, header=None, sep="\t")

        min_x = int(float(affine.loc["min_x"].item()))
        min_y = int(float(affine.loc["min_y"].item()))
        size_x = int(float(affine.loc["size_x"].item()))
        size_y = int(float(affine.loc["size_y"].item()))

        dapi = dapi[min_y : min_y + size_y, min_x : min_x + size_x]

    dapi_h = dapi.shape[0]
    dapi_w = dapi.shape[1]
    print(f"DAPI shape h: {dapi_h} w: {dapi_w}")

    # Process patch-wise if too large
    if (
        dapi_h > config.nuclei.max_height
        or dapi_w > config.nuclei.max_width
        or config.affine.scale_pix_x != 1.0
        or config.affine.scale_pix_y != 1.0
    ):
        if config.nuclei.max_height is None:
            max_height = dapi_h
        else:
            max_height = config.nuclei.max_height if config.nuclei.max_height < dapi_h else dapi_h
        if config.nuclei.max_width is None:
            max_width = dapi_w
        else:
            max_width = config.nuclei.max_width if config.nuclei.max_width < dapi_w else dapi_w

        print(f"Segmenting DAPI patches h: {max_height} w: {max_width}")

        # Coordinates of patches
        h_coords, _ = get_patches_coords(dapi_h, max_height)
        w_coords, _ = get_patches_coords(dapi_w, max_width)
        hw_coords = [(hs, he, ws, we) for (hs, he) in h_coords for (ws, we) in w_coords]

        # Original patch sizes
        h_patch_sizes = [he - hs for (hs, he) in h_coords]
        w_patch_sizes = [we - ws for (ws, we) in w_coords]

        # Determine the resized patch sizes
        rh_patch_sizes = [round(y * config.affine.scale_pix_y) for y in h_patch_sizes]
        rw_patch_sizes = [round(x * config.affine.scale_pix_x) for x in w_patch_sizes]
        rhw_patch_sizes = [
            (hsize, wsize) for hsize in rh_patch_sizes for wsize in rw_patch_sizes
        ]

        # Determine the resized patch starting coordinates
        rh_coords = [sum(rh_patch_sizes[:i]) for i, x in enumerate(rh_patch_sizes)]
        rw_coords = [sum(rw_patch_sizes[:i]) for i, x in enumerate(rw_patch_sizes)]
        rhw_coords = [(h, w) for h in rh_coords for w in rw_coords]

        # Sum up the new sizes to get the final size of the resized DAPI
        rh_dapi = sum(rh_patch_sizes)
        rw_dapi = sum(rw_patch_sizes)
        rdapi = np.zeros((rh_dapi, rw_dapi), dtype=dapi.dtype)
        nuclei = np.zeros((rh_dapi, rw_dapi), dtype=np.uint32)
        print(f"Nuclei image h: {rh_dapi} w: {rw_dapi}")

        # Divide into patches
        n_patches = len(hw_coords)
        total_n = 0

        for patch_i in tqdm(range(n_patches)):
            (hs, he, ws, we) = hw_coords[patch_i]
            (hsize, wsize) = rhw_patch_sizes[patch_i]
            (h, w) = rhw_coords[patch_i]

            patch = dapi[hs:he, ws:we]

            patch_resized = resize_dapi(patch, hsize, wsize)
            rdapi[h : h + hsize, w : w + wsize] = patch_resized

            # Segment nuclei in each patch and place into final segmentation with unique ID
            patch_nuclei = segment_dapi(patch_resized, config.nuclei.diameter, config.nuclei.use_cpu)
            nuclei_mask = np.where(patch_nuclei > 0, 1, 0)
            nuclei[h : h + hsize, w : w + wsize] = patch_nuclei + total_n * nuclei_mask
            unique_ids = np.unique(patch_nuclei)
            total_n += unique_ids.max()

        # Save resized DAPI
        fp_rdapi = os.path.join(dir_dataset, config.files.fp_rdapi)
        tifffile.imwrite(fp_rdapi, rdapi, photometric="minisblack")

    else:
        print("Segmenting whole DAPI")
        nuclei = segment_dapi(dapi)

    print(f"Finished segmenting, found {len(np.unique(nuclei))-1} nuclei")

    # Save nuclei segmentation
    fp_nuclei = os.path.join(dir_dataset, config.files.fp_nuclei)
    tifffile.imwrite(fp_nuclei, nuclei.astype(np.uint32), photometric="minisblack")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_dir", type=str, help="path to config"
    )

    args = parser.parse_args()
    config = load_config(args.config_dir)

    segment_nuclei(config)
