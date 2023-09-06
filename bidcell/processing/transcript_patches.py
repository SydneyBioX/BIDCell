import argparse
import glob
import os
import re

import h5py
import natsort
import numpy as np
from tqdm import tqdm

from ..config import Config, load_config


def generate_patches(config: Config):
    """
    Divides transcriptomic maps of all genes into patches for input to the CNN

    """
    dir_dataset = os.path.join(config.files.data_dir, config.files.dir_out_maps)

    patch_size = config.model_params.patch_size
    shift = [0, int(patch_size / 2)]

    fp_maps = glob.glob(dir_dataset + "/all_genes_*.hdf5")
    fp_maps = natsort.natsorted(fp_maps)

    for fp in fp_maps:
        print(f"Processing {fp}")
        h5f = h5py.File(fp, "r")
        sst = h5f["data"][:]
        h5f.close()
        print("Loaded gene expr maps from %s" % fp)

        # hs, he, ws, we
        map_h = sst.shape[0]
        map_w = sst.shape[1]
        map_coords = [
            int(x)
            for x in re.findall(r"\d+", os.path.basename(fp).replace(".hdf5", ""))
        ]
        print(map_coords)

        h_lim = sst.shape[0]
        w_lim = sst.shape[1]

        # If map contains blank border
        if (map_coords[1] - map_coords[0]) < map_h:
            h_lim = map_coords[1] - map_coords[0]
        if (map_coords[3] - map_coords[2]) < map_w:
            w_lim = map_coords[3] - map_coords[2]

        # print(h_lim, w_lim)

        for shift_patches in shift:
            print("Shift by %d" % shift_patches)

            dir_output = os.path.join(
                dir_dataset,
                config.files.dir_patches
                + "%dx%d_shift_%d" % (patch_size, patch_size, shift_patches),
            )
            if not os.path.exists(dir_output):
                os.makedirs(dir_output)

            # Get coordinates of non-overlapping patches
            if shift_patches == 0:
                h_starts = list(np.arange(0, h_lim - patch_size, patch_size))
                w_starts = list(np.arange(0, w_lim - patch_size, patch_size))

                # Include remainder patches on
                h_starts.append(h_lim - patch_size)
                w_starts.append(w_lim - patch_size)

            else:
                h_starts = list(
                    np.arange(shift_patches, h_lim - patch_size, patch_size)
                )
                w_starts = list(
                    np.arange(shift_patches, w_lim - patch_size, patch_size)
                )

            coords_starts = [(x, y) for x in h_starts for y in w_starts]
            print(f"{len(coords_starts)} patches")

            # Get patches and save
            for h, w in tqdm(coords_starts):
                patch = sst[h : h + patch_size, w : w + patch_size, :]

                fp_output = f"{dir_output}/{h+map_coords[0]}_{w+map_coords[2]}.hdf5"

                h = h5py.File(fp_output, "w")
                _ = h.create_dataset("data", data=patch, dtype=np.uint8)
                h.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    generate_patches(config)
