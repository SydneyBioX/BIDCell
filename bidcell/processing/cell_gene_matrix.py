import argparse
import glob
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from .utils import get_n_processes, get_patches_coords
from ..config import Config, load_config

np.seterr(divide="ignore", invalid="ignore")


def process_chunk(
    chunk, output_dir, cell_ids_unique, col_names, seg_map, x_col, y_col, gene_col
):
    """Extract cell expression profiles"""

    df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
    df_out["cell_id"] = cell_ids_unique.copy()

    chunk_id = chunk.index[0]

    for index_row, row in chunk.iterrows():
        gene = row[gene_col]
        w_loc = row[x_col]
        h_loc = row[y_col]

        seg_val = seg_map[h_loc, w_loc]
        if seg_val > 0:
            df_out.loc[seg_val, gene] += 1

    df_out.to_csv(output_dir + "/" + "chunk_%d.csv" % chunk_id)


def process_chunk_meta(
    matrix, fp_output, seg_map_mi, col_names_coords, scale_pix_x, scale_pix_y
):
    """Compute cell locations and sizes"""

    chunk_id = matrix[0, 0]
    output = np.zeros((matrix.shape[0], len(col_names_coords)))
    output[:, 0] = matrix[:, 0].copy()
    output[:, 4:] = matrix[:, 1:].copy()

    # Convert to pixel resolution
    for cur_i, cell_id in enumerate(output[:, 0]):
        if cell_id > 0:
            try:
                # cell_centroid_x and cell_centroid_y
                coords = np.where(seg_map_mi == cell_id)
                x_points = coords[1]
                y_points = coords[0]
                centroid_x = sum(x_points) / len(x_points)
                centroid_y = sum(y_points) / len(y_points)
                output[cur_i, 1] = centroid_x * scale_pix_x
                output[cur_i, 2] = centroid_y * scale_pix_y

                # cell_size
                output[cur_i, 3] = len(coords[0]) / (scale_pix_x * scale_pix_y)
            except Exception:
                output[cur_i, 1] = -1
                output[cur_i, 2] = -1
                output[cur_i, 3] = -1

    # Save as csv
    df_split = pd.DataFrame(
        output, index=list(range(output.shape[0])), columns=col_names_coords
    )
    df_split.to_csv(fp_output + "%d.csv" % chunk_id, index=False)


def transform_locations(df_expr, col, scale, shift=0):
    """Scale transcripts to pixel resolution of the platform"""
    print(f"Transforming {col}")
    df_expr[col] = df_expr[col].div(scale).round().astype(int).sub(shift)
    return df_expr


def read_expr_csv(fp):
    try:
        print("Reading filtered transcripts")
        return pd.read_csv(fp)
    except Exception:
        sys.exit(f"Cannot read {fp}")


def make_cell_gene_mat(config: Config, is_cell: bool, timestamp: str | None = None):
    dir_dataset = config.files.data_dir
    dir_cgm = config.files.dir_cgm

    if is_cell is False:
        output_dir = os.path.join(dir_dataset, dir_cgm, "nuclei")
    else:
        output_dir = os.path.join(dir_dataset, dir_cgm, timestamp)

    fp_transcripts_processed = os.path.join(
        dir_dataset, config.files.fp_transcripts_processed
    )

    fp_gene_names = os.path.join(dir_dataset, config.files.fp_gene_names)

    if is_cell is False:
        fp_seg = os.path.join(dir_dataset, config.files.fp_nuclei)
    else:
        fp_seg_name = [
            "epoch_"
            + str(config.testing_params.test_epoch)
            + "_step_"
            + str(config.testing_params.test_step)
            + "_connected.tif"
        ]
        fp_seg = os.path.join(
            config.files.data_dir,
            "model_outputs",
            timestamp,
            config.experiment_dirs.test_output_dir,
            "".join(fp_seg_name),
        )

    # Column names in the transcripts csv
    x_col = config.transcripts.x_col
    y_col = config.transcripts.y_col
    gene_col = config.transcripts.gene_col

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_map_mi = tifffile.imread(fp_seg)
    height = seg_map_mi.shape[0]
    width = seg_map_mi.shape[1]

    cell_ids_unique = np.unique(seg_map_mi.reshape(-1))
    cell_ids_unique = cell_ids_unique[1:]
    n_cells = len(cell_ids_unique)
    print("Number of cells " + str(n_cells))

    with open(fp_gene_names) as file:
        gene_names = [line.rstrip() for line in file]

    col_names = ["cell_id"] + gene_names

    # Divide the dataframe into chunks for multiprocessing
    n_processes = get_n_processes(config.cpus)
    print(f"Number of splits for multiprocessing: {n_processes}")

    # Scale factor to pixel resolution of platform
    # read in affine
    # extract scale_x and scale_y
    # divide by (scale_x*pixel resolution) (microns per pixel)
    # affine = pd.read_csv(fp_affine, index_col=0, header=None, sep='\t')
    # scale_x_tr = float(affine.loc["scale_x"].item())
    # scale_y_tr = float(affine.loc["scale_y"].item())
    # scale_pix_x = (scale_x_tr*config.affine.scale_pix_x)
    # scale_pix_y = (scale_y_tr*config.affine.scale_pix_y)
    scale_pix_x = config.affine.scale_pix_x
    scale_pix_y = config.affine.scale_pix_y

    if not os.path.exists(output_dir + "/" + config.files.fp_expr):
        # Rescale to pixel size
        height_pix = np.round(height / config.affine.scale_pix_y).astype(int)
        width_pix = np.round(width / config.affine.scale_pix_x).astype(int)

        seg_map = cv2.resize(
            seg_map_mi.astype(np.int32),
            (width_pix, height_pix),
            interpolation=cv2.INTER_NEAREST,
        )
        print("Segmentation map pixel size: ", seg_map.shape)
        fp_rescaled_seg = output_dir + "/rescaled.tif"
        print("Saving temporary resized segmentation")
        tifffile.imwrite(
            fp_rescaled_seg, seg_map.astype(np.uint32), photometric="minisblack"
        )

        df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
        df_out["cell_id"] = cell_ids_unique.copy()

        # Divide into patches for large datasets that exceed memory capacity
        if (height_pix + width_pix) > config.cgm_params.max_sum_hw:
            patch_h = int(config.cgm_params.max_sum_hw / 2)
            patch_w = config.cgm_params.max_sum_hw - patch_h
        else:
            patch_h = height_pix
            patch_w = width_pix

        h_coords, _ = get_patches_coords(height_pix, patch_h)
        w_coords, _ = get_patches_coords(width_pix, patch_w)
        hw_coords = [(hs, he, ws, we) for (hs, he) in h_coords for (ws, we) in w_coords]

        print("Extracting cell expressions")
        for hs, he, ws, we in tqdm(hw_coords):
            print(f"Patch H {hs}:{he}, W {ws}:{we}")
            seg_map = tifffile.imread(fp_rescaled_seg)[hs:he, ws:we]
            print(seg_map.shape)

            df_expr = read_expr_csv(fp_transcripts_processed)
            print(
                df_expr[x_col].min(),
                df_expr[x_col].max(),
                df_expr[y_col].min(),
                df_expr[y_col].max(),
            )

            df_expr = transform_locations(df_expr, x_col, scale_pix_x)
            df_expr = transform_locations(df_expr, y_col, scale_pix_y)

            df_expr = df_expr[
                (df_expr[x_col].between(ws, we - 1))
                & (df_expr[y_col].between(hs, he - 1))
            ]
            print(
                df_expr[x_col].min(),
                df_expr[x_col].max(),
                df_expr[y_col].min(),
                df_expr[y_col].max(),
            )

            df_expr = transform_locations(df_expr, x_col, 1, ws)
            df_expr = transform_locations(df_expr, y_col, 1, hs)
            print(
                df_expr[x_col].min(),
                df_expr[x_col].max(),
                df_expr[y_col].min(),
                df_expr[y_col].max(),
            )

            df_expr.reset_index(drop=True, inplace=True)

            df_expr_splits = np.array_split(df_expr, n_processes)
            processes = []

            print("Extracting cell-gene matrix chunks")
            for chunk in df_expr_splits:
                p = mp.Process(
                    target=process_chunk,
                    args=(
                        chunk,
                        output_dir,
                        cell_ids_unique,
                        col_names,
                        seg_map,
                        x_col,
                        y_col,
                        gene_col,
                    ),
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            print("Combining cell-gene matrix chunks")

            fp_chunks = glob.glob(output_dir + "/chunk_*.csv")
            for fpc in fp_chunks:
                df_i = pd.read_csv(fpc, index_col=0)
                df_out.iloc[:, 1:] = df_out.iloc[:, 1:].add(df_i.iloc[:, 1:])

            df_out.to_csv(output_dir + "/" + config.files.fp_expr)

            # Clean up
            for fpc in fp_chunks:
                os.remove(fpc)

        print("Obtained cell-gene matrix")
        os.remove(fp_rescaled_seg)
        del seg_map
        del df_expr

    else:
        df_out = pd.read_csv(output_dir + "/" + config.files.fp_expr, index_col=0)

    # if is_cell:
    #     print("Computing cell locations and sizes")

    #     matrix_all = df_out.to_numpy().astype(np.float32)
    #     matrix_all_splits = np.array_split(matrix_all, n_processes)
    #     processes = []

    #     fp_output = output_dir + "/cell_outputs_"
    #     col_names_coords = [
    #         "cell_id",
    #         "cell_centroid_x",
    #         "cell_centroid_y",
    #         "cell_size",
    #     ] + gene_names

    #     for chunk in matrix_all_splits:
    #         p = mp.Process(
    #             target=process_chunk_meta,
    #             args=(
    #                 chunk,
    #                 fp_output,
    #                 seg_map_mi,
    #                 col_names_coords,
    #                 scale_pix_x,
    #                 scale_pix_y,
    #             ),
    #         )
    #         processes.append(p)
    #         p.start()

    #     for p in processes:
    #         p.join()

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")
    parser.add_argument(
        "--is_cell", type=bool, help="whether to segment cells or nuclei"
    )

    args = parser.parse_args()
    config = load_config(args.config_dir)

    make_cell_gene_mat(config, args.is_cell)
