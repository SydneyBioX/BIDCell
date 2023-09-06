import argparse
import glob
import multiprocessing as mp
import os
import random
import re
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from ..config import load_config, Config


def get_n_processes(n_processes):
    """Number of CPUs for multiprocessing"""
    if n_processes is None:
        return mp.cpu_count()
    else:
        return n_processes if n_processes <= mp.cpu_count() else mp.cpu_count()


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def get_exp_dir(config):
    if config.files.dir_id == "last":
        folders = next(os.walk("model_outputs"))[1]
        folders = sorted_alphanumeric(folders)
        folder_last = folders[-1]
        dir_id = folder_last.replace("\\", "/")
    else:
        dir_id = config.files.dir_id

    return dir_id


def postprocess_connect(img, nuclei):
    cell_ids = np.unique(img)
    cell_ids = cell_ids[1:]

    random.shuffle(cell_ids)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Touch diagonally = same object
    s = ndi.generate_binary_structure(2, 2)

    final = np.zeros(img.shape, dtype=np.uint32)

    for i in cell_ids:
        i_mask = np.where(img == i, 1, 0).astype(np.uint8)

        connected_mask = cv2.dilate(i_mask, kernel, iterations=2)
        connected_mask = cv2.erode(connected_mask, kernel, iterations=2)

        # Add nucleus as predicted by cellpose
        nucleus_mask = np.where(nuclei == i, 1, 0).astype(np.uint8)

        connected_mask = connected_mask + nucleus_mask
        connected_mask[connected_mask > 0] = 1

        unique_ids, num_ids = ndi.label(connected_mask, structure=s)
        if num_ids > 1:
            # The first element is always 0 (background)
            unique, counts = np.unique(unique_ids, return_counts=True)

            # Ensure counts in descending order
            counts, unique = (list(t) for t in zip(*sorted(zip(counts, unique))))
            counts.reverse()
            unique.reverse()
            counts = np.array(counts)
            unique = np.array(unique)

            no_overlap = False

            # Rarely, the nucleus is not the largest segment
            for i_part in range(1, len(counts)):
                if i_part > 1:
                    no_overlap = True  # TODO: Helen, check this!
                largest = unique[np.argmax(counts[i_part:]) + i_part]
                connected_mask = np.where(unique_ids == largest, 1, 0)
                # Break if current largest region overlaps nucleus
                if np.sum(connected_mask * nucleus_mask) > 0.5:
                    break

            # Close holes on largest section
            filled_mask = ndi.binary_fill_holes(connected_mask).astype(int)

        else:
            filled_mask = ndi.binary_fill_holes(connected_mask).astype(int)

        final = np.where(filled_mask > 0, i, final)

    final = np.where(nuclei > 0, nuclei, final)
    return final


def remove_islands(img, nuclei):
    cell_ids = np.unique(img)
    cell_ids = cell_ids[1:]

    random.shuffle(cell_ids)

    # Touch diagonally = same object
    s = ndi.generate_binary_structure(2, 2)

    final = np.zeros(img.shape, dtype=np.uint32)

    for i in cell_ids:
        i_mask = np.where(img == i, 1, 0).astype(np.uint8)

        nucleus_mask = np.where(nuclei == i, 1, 0).astype(np.uint8)

        # Number of blobs belonging to cell
        unique_ids, num_blobs = ndi.label(i_mask, structure=s)
        if num_blobs > 1:
            # Keep the blob with max overlap to nucleus
            amount_overlap = np.zeros(num_blobs)

            for i_blob in range(1, num_blobs + 1):
                blob = np.where(unique_ids == i_blob, 1, 0)
                amount_overlap[i_blob - 1] = np.sum(blob * nucleus_mask)
            blob_keep = np.argmax(amount_overlap) + 1

            final_mask = np.where(unique_ids == blob_keep, 1, 0)

        else:
            blob_size = np.count_nonzero(i_mask)
            if blob_size > 2:
                final_mask = i_mask.copy()
            else:
                final_mask = i_mask * 0

        final_mask = ndi.binary_fill_holes(final_mask).astype(int)

        final = np.where(final_mask > 0, i, final)

    return final


def process_chunk(chunk, patch_size, img_whole, nuclei_img, output_dir):
    for index in range(len(chunk)):
        coords = chunk[index]
        coords_x1 = coords[0]
        coords_y1 = coords[1]
        coords_x2 = coords_x1 + patch_size
        coords_y2 = coords_y1 + patch_size

        img = img_whole[coords_x1:coords_x2, coords_y1:coords_y2]

        nuclei = nuclei_img[coords_x1:coords_x2, coords_y1:coords_y2]

        output_fp = output_dir + "%d_%d.tif" % (coords_x1, coords_y1)

        # print('Filling holes')
        filled = postprocess_connect(img, nuclei)

        # print('Removing islands')
        final = remove_islands(filled, nuclei)

        tifffile.imwrite(output_fp, final.astype(np.uint32), photometric="minisblack")

        cell_ids = np.unique(final)[1:]

        # Visualise cells with random colours
        n_cells_ids = len(cell_ids)
        cell_ids_rand = np.arange(1, n_cells_ids + 1)
        random.shuffle(cell_ids_rand)

        keep_mask = np.isin(nuclei, cell_ids)
        nuclei = np.where(keep_mask, nuclei, 0)
        keep_mask = np.isin(img, cell_ids)
        img = np.where(keep_mask, nuclei, 0)

        dictionary = dict(zip(cell_ids, cell_ids_rand))
        dictionary[0] = 0
        nuclei_mapped = np.copy(nuclei)
        img_mapped = np.copy(img)
        final_mapped = np.copy(final)

        nuclei_mapped = np.vectorize(dictionary.get)(nuclei)
        img_mapped = np.vectorize(dictionary.get)(img)
        final_mapped = np.vectorize(dictionary.get)(final)

        fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(nuclei_mapped, cmap=plt.cm.gray)
        ax[0].set_title("Nuclei")
        ax[1].imshow(img_mapped, cmap=plt.cm.nipy_spectral)
        ax[1].set_title("Original")
        ax[2].imshow(final_mapped, cmap=plt.cm.nipy_spectral)
        ax[2].set_title("Processed")
        for a in ax:
            a.set_axis_off()
        fig.tight_layout()
        # plt.show()

        fig.savefig(output_fp.replace("tif", "png"), dpi=300)
        plt.close(fig)


def combine(config, dir_id, patch_size, nuclei_img):
    """
    Combine the patches previously output by the connect function
    """

    fp_dir = dir_id + "epoch_%d_step_%d_connected" % (
        config.testing_params.test_epoch,
        config.testing_params.test_step,
    )
    fp_unconnected = dir_id + "epoch_%d_step_%d.tif" % (
        config.testing_params.test_epoch,
        config.testing_params.test_step,
    )

    dl_pred = tifffile.imread(fp_unconnected)
    height = dl_pred.shape[0]
    width = dl_pred.shape[1]

    seg_final = np.zeros((height, width), dtype=np.uint32)

    fp_seg = glob.glob(fp_dir + "/*.tif", recursive=True)

    sample = tifffile.imread(fp_seg[0])
    patch_h = sample.shape[0]
    patch_w = sample.shape[1]

    cell_ids = []

    for fp in fp_seg:
        patch = tifffile.imread(fp)

        patch_ids = np.unique(patch)
        patch_ids = patch_ids[patch_ids != 0]
        cell_ids.extend(patch_ids)

        fp_coords = os.path.basename(fp).split(".")[0]
        fp_x = int(fp_coords.split("_")[0])
        fp_y = int(fp_coords.split("_")[1])

        # Place into appropriate location
        seg_final[fp_x : fp_x + patch_h, fp_y : fp_y + patch_w] = patch[:]

    # If cell is split by windowing, keep component with nucleus
    count_ids = Counter(cell_ids)
    windowed_ids = [k for k, v in count_ids.items() if v > 1]

    # Check along borders
    h_starts = list(np.arange(0, height - patch_size, patch_size))
    w_starts = list(np.arange(0, width - patch_size, patch_size))
    h_starts.append(height - patch_size)
    w_starts.append(width - patch_size)

    # Mask along grid
    h_starts_wide = []
    w_starts_wide = []
    for i in range(-10, 11):
        h_starts_wide.extend([x + i for x in h_starts])
        w_starts_wide.extend([x + i for x in w_starts])

    mask = np.zeros(seg_final.shape)
    mask[h_starts_wide, :] = 1
    mask[:, w_starts_wide] = 1

    masked = mask * seg_final
    masked_ids = np.unique(masked)[1:]

    # IDs to check for split bodies
    to_check_ids = list(set(masked_ids) & set(windowed_ids))

    return seg_final, to_check_ids


def process_check_splits(config, dir_id, nuclei_img, seg_final, chunk_ids):
    """
    Check and fix cells split by windowing
    """

    chunk_seg = np.zeros(seg_final.shape, dtype=np.uint32)

    # Touch diagonally = same object
    s = ndi.generate_binary_structure(2, 2)

    for i in chunk_ids:
        i_mask = np.where(seg_final == i, 1, 0).astype(np.uint8)

        # Number of blobs belonging to cell
        unique_ids, num_blobs = ndi.label(i_mask, structure=s)

        # Bounding box
        bb = np.argwhere(unique_ids)
        (ystart, xstart), (ystop, xstop) = bb.min(0), bb.max(0) + 1
        unique_ids_crop = unique_ids[ystart:ystop, xstart:xstop]

        nucleus_mask = np.where(nuclei_img == i, 1, 0).astype(np.uint8)
        nucleus_mask = nucleus_mask[ystart:ystop, xstart:xstop]

        if num_blobs > 1:
            # Keep the blob with max overlap to nucleus
            amount_overlap = np.zeros(num_blobs)

            for i_blob in range(1, num_blobs + 1):
                blob = np.where(unique_ids_crop == i_blob, 1, 0)
                amount_overlap[i_blob - 1] = np.sum(blob * nucleus_mask)
            blob_keep = np.argmax(amount_overlap) + 1

            # Put into final segmentation
            final_mask = np.where(unique_ids_crop == blob_keep, 1, 0)

            # seg_final = np.where(seg_final == i, 0, seg_final)

            # # Double check the few outliers
            # unique_ids_2, num_blobs_2 = ndi.label(final_mask, structure=s)
            # if num_blobs_2 > 1:
            #     # Keep largest
            #     blob_keep_2 = np.argmax(np.bincount(unique_ids_2)[1:]) + 1
            #     final_mask = np.where(unique_ids_2 == blob_keep_2, 1, 0)

            chunk_seg[ystart:ystop, xstart:xstop] = np.where(
                final_mask == 1, i, chunk_seg[ystart:ystop, xstart:xstop]
            )

        else:
            chunk_seg = np.where(i_mask == 1, i, chunk_seg)

    tifffile.imwrite(
        dir_id + "/" + str(chunk_ids[0]) + "_checked_splits.tif",
        chunk_seg,
        photometric="minisblack",
    )


def postprocess_predictions(config: Config, dir_id: str):
    dir_id = config.files.data_dir + "/model_outputs/" + dir_id + "/test_output/"

    pred_fp = dir_id + "epoch_%d_step_%d.tif" % (
        config.testing_params.test_epoch,
        config.testing_params.test_step,
    )
    output_dir = dir_id + "epoch_%d_step_%d_connected/" % (
        config.testing_params.test_epoch,
        config.testing_params.test_step,
    )

    nucleus_fp = os.path.join(config.files.data_dir, config.files.fp_nuclei)
    nuclei_img = tifffile.imread(nucleus_fp)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_whole = tifffile.imread(pred_fp)

    smallest_dim = np.min(img_whole.shape)
    if config.postprocess.patch_size_mp < smallest_dim:
        patch_size = config.postprocess.patch_size_mp
    else:
        patch_size = smallest_dim

    h_starts = list(np.arange(0, img_whole.shape[0] - patch_size, patch_size))
    w_starts = list(np.arange(0, img_whole.shape[1] - patch_size, patch_size))
    h_starts.append(img_whole.shape[0] - patch_size)
    w_starts.append(img_whole.shape[1] - patch_size)
    coords_starts = [(x, y) for x in h_starts for y in w_starts]
    print("%d patches available" % len(coords_starts))

    # num_processes = mp.cpu_count()
    num_processes = get_n_processes(config.cpus)
    print("Num multiprocessing splits: %d" % num_processes)

    coords_splits = np.array_split(coords_starts, num_processes)
    processes = []

    print("Processing...")

    for chunk in coords_splits:
        p = mp.Process(
            target=process_chunk,
            args=(chunk, patch_size, img_whole, nuclei_img, output_dir),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Combining results")
    seg_final, to_check_ids = combine(config, dir_id, patch_size, nuclei_img)
    # print(len(np.unique(seg_final)), len(to_check_ids))

    ids_splits = np.array_split(to_check_ids, num_processes)
    processes = []

    for chunk_ids in ids_splits:
        p = mp.Process(
            target=process_check_splits,
            args=(config, dir_id, nuclei_img, seg_final, chunk_ids),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    check_mask = np.isin(seg_final, to_check_ids)
    seg_final = np.where(check_mask == 1, 0, seg_final)

    fp_checked_splits = glob.glob(dir_id + "/*_checked_splits.tif", recursive=True)
    for fp in fp_checked_splits:
        checked_split = tifffile.imread(fp)
        seg_final = np.where(checked_split > 0, checked_split, seg_final)
        os.remove(fp)

    seg_final = np.where(nuclei_img > 0, nuclei_img, seg_final)

    fp_dir = dir_id + "epoch_%d_step_%d_connected" % (
        config.testing_params.test_epoch,
        config.testing_params.test_step,
    )
    fp_output_seg = fp_dir + ".tif"
    print("Saved segmentation to %s" % fp_output_seg)
    tifffile.imwrite(
        fp_output_seg, seg_final.astype(np.uint32), photometric="minisblack"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    postprocess_predictions(config)
