import argparse
import glob
import os
import re
import sys

import natsort
import numpy as np
import tifffile
from PIL import Image
from ..config import Config, load_config


def check_pattern(string, pattern):
    """Check that a string contains a pattern"""
    pattern = pattern.replace("#", r"\d")  # Replace "#" with "\d" to match any digit
    match = re.search(pattern, string)
    # print(match, match.group())

    if match:
        return True  # Pattern found in the string
    else:
        return False  # Pattern not found in the string


def check_shape_imgs(fp, target_h, target_w):
    """Check that an image (given its file path) has the same shape as target"""
    img = Image.open(fp)
    if img.size == (target_w, target_h):
        return True
    else:
        return False


def check_images_meet_criteria(fp_list, boolean_list, msg):
    """Check if any file in a list of paths doesn't meet criteria"""
    wrong = [i for i, x in enumerate(boolean_list) if not x]
    if len(wrong) > 0:
        sys.exit(f"{msg}: {[fp_list[i] for i in wrong]}")


def get_string_with_pattern(number, pattern):
    """1 # means F1..F10, etc. >1 # means F001..F010, etc"""
    num_hash = pattern.count("#")
    no_hash = pattern.replace("#", "")

    if num_hash == 1:
        return no_hash + str(number)
    elif num_hash >= len(str(number)):
        padded = str(number).zfill(num_hash)
        return no_hash + str(padded)
    else:
        sys.exit(f"Number requires more characters than {pattern}")


def read_dapi(fp, channel_first, channel_dapi):
    """Reads DAPI image or channel from file"""
    dapi = tifffile.imread(fp)

    if len(dapi.shape) > 2:
        if channel_first:
            dapi = dapi[channel_dapi, :, :]
        else:
            dapi = dapi[:, :, channel_dapi]

    return dapi


def stitch_nuclei(config: Config):
    dir_dataset = config.files.data_dir

    if not config.nuclei_fovs.dir_dapi:
        dir_dapi = dir_dataset
    else:
        dir_dapi = config.nuclei_fovs.dir_dapi

    ext_pat = "".join(
        "[%s%s]" % (e.lower(), e.upper()) for e in config.nuclei_fovs.ext_dapi
    )
    fp_dapi_list = glob.glob(os.path.join(dir_dapi, "*." + ext_pat))
    fp_dapi_list = natsort.natsorted(fp_dapi_list)

    sample = tifffile.imread(fp_dapi_list[0])
    fov_shape = sample.shape

    # Error if multi-channel image and channel is not specified
    if len(fov_shape) > 2:
        if config.nuclei_fovs.channel_first:
            print(f"Channel axis first, DAPI channel {config.nuclei_fovs.channel_dapi}")
            fov_h = sample.shape[1]
            fov_w = sample.shape[2]
        else:
            print(f"Channel axis last, DAPI channel {config.nuclei_fovs.channel_dapi}")
            fov_h = sample.shape[0]
            fov_w = sample.shape[1]

    fov_dtype = sample.dtype

    # Error if patterns in file path names not found
    found_f = [check_pattern(s, config.nuclei_fovs.pattern_f) for s in fp_dapi_list]
    check_images_meet_criteria(fp_dapi_list, found_f, "FOV string pattern not found in")

    if config.nuclei_fovs.pattern_z is not None:
        found_z = [check_pattern(s, config.nuclei_fovs.pattern_z) for s in fp_dapi_list]
        check_images_meet_criteria(
            fp_dapi_list, found_z, "Z slice string pattern not found in"
        )

    # Check shape the same as sample
    match_shape = [check_shape_imgs(s, fov_h, fov_w) for s in fp_dapi_list]
    check_images_meet_criteria(fp_dapi_list, match_shape, "Different image shape for")

    # Locations of each FOV in the whole image
    n_fov = config.nuclei_fovs.n_fov
    if config.nuclei_fovs.row_major:
        order = np.arange(
            config.nuclei_fovs.n_fov_h * config.nuclei_fovs.n_fov_w
        ).reshape((config.nuclei_fovs.n_fov_h, config.nuclei_fovs.n_fov_w))
    else:
        order = np.arange(
            config.nuclei_fovs.n_fov_h * config.nuclei_fovs.n_fov_w
        ).reshape((config.nuclei_fovs.n_fov_h, config.nuclei_fovs.n_fov_w), order="F")

    # Arrangement of the FOVs - default is ul
    if config.nuclei_fovs.start_corner == "ur":
        order = np.flip(order, 1)
    elif config.nuclei_fovs.start_corner == "bl":
        order = np.flip(order, 0)
    elif config.nuclei_fovs.start_corner == "br":
        order = np.flip(order, (0, 1))

    print("FOV ordering")
    print(order)

    stitched = np.zeros(
        (fov_h * config.nuclei_fovs.n_fov_h, fov_w * config.nuclei_fovs.n_fov_w),
        dtype=fov_dtype,
    )

    for i_fov in range(n_fov):
        coord = np.where(order == i_fov)
        h_idx = coord[0][0]
        w_idx = coord[1][0]
        h_start = h_idx * fov_h
        w_start = w_idx * fov_w
        h_end = h_start + fov_h
        w_end = w_start + fov_w

        fov_num = i_fov + config.nuclei_fovs.min_fov

        # All files for FOV
        pattern_fov = get_string_with_pattern(fov_num, config.nuclei_fovs.pattern_f)
        print(pattern_fov)
        found_fov = [check_pattern(s, pattern_fov) for s in fp_dapi_list]
        fp_stack_fov = [fp_dapi_list[i] for i, x in enumerate(found_fov) if x]

        # Take MIP - or z level
        if config.nuclei_fovs.mip:
            dapi_stack = np.zeros((len(fp_stack_fov), fov_h, fov_w), dtype=fov_dtype)
            for i, fp in enumerate(fp_stack_fov):
                dapi_stack[i, :, :] = read_dapi(
                    fp,
                    config.nuclei_fovs.channel_first,
                    config.nuclei_fovs.channel_dapi,
                )

            fov_img = np.max(dapi_stack, axis=0)

        else:
            # Find z level slice for FOV
            pattern_slice = get_string_with_pattern(
                config.nuclei_fovs.z_level, config.nuclei_fovs.pattern_z
            )
            print(pattern_slice)
            found_slice = [check_pattern(s, pattern_slice) for s in fp_stack_fov]
            found_slice_idx = [i for i, x in enumerate(found_slice) if x]
            if len(found_slice_idx) > 1:
                sys.exit(
                    f"Found {len(found_slice_idx)} files with {pattern_slice} for FOV {fov_num}"
                )

            print(fp_stack_fov[found_slice_idx[0]])
            fov_img = read_dapi(
                fp_stack_fov[found_slice_idx[0]],
                config.nuclei_fovs.channel_first,
                config.nuclei_fovs.channel_dapi,
            )

        # Flip
        if config.nuclei_fovs.flip_ud:
            fov_img = np.flip(fov_img, 0)

        # Place into appropriate location in stitched image
        stitched[h_start:h_end, w_start:w_end] = fov_img.copy()

    # Save
    fp_output = os.path.join(dir_dataset, "dapi_stitched.tif")
    tifffile.imwrite(fp_output, stitched, photometric="minisblack")
    print(f"Saved {fp_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    stitch_nuclei(config)
