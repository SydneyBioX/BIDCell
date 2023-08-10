import tifffile
import pandas as pd
import numpy as np
import h5py
import os
import natsort
import argparse
from tqdm import tqdm 
import multiprocessing as mp
import pathlib
import glob
import re
import warnings
import sys
import csv
from utils import get_patches_coords, get_n_processes

def process_gene_chunk(gene_chunk, df_patch, img_height, img_width, dir_output, hs, ws,
                       gene_col, x_col, y_col, counts_col):
    
    # print(gene_chunk)
    for i_fe, fe in enumerate(gene_chunk):
        # print(fe)
        df_fe = df_patch.loc[df_patch[gene_col] == fe]
        map_fe = np.zeros((img_height, img_width))
        # print(map_fe.shape)

        if counts_col == None:
            for idx in df_fe.index:
                idx_x = np.round(df_patch.iloc[idx][x_col]).astype(int)
                idx_y = np.round(df_patch.iloc[idx][y_col]).astype(int)

                map_fe[idx_y, idx_x] += 1

        else:
            for idx in df_fe.index:
                idx_x = np.round(df_patch.iloc[idx][x_col]).astype(int)
                idx_y = np.round(df_patch.iloc[idx][y_col]).astype(int)
                idx_counts = df_patch.iloc[idx][counts_col]

                map_fe[idx_y, idx_x] += idx_counts

        # print(map_fe.shape)

        fp_fe_map = f"{dir_output}/{fe}_{hs}_{ws}.tif"
        # print(fp_fe_map)
        tifffile.imwrite(fp_fe_map, map_fe.astype(np.uint8), photometric='minisblack')


def stitch_patches(dir_patches, fp_pattern):
    """Stitches together the patches of summed genes and saves as new tif"""
    fp_patches = glob.glob(dir_patches + '/' + fp_pattern)
    fp_patches = natsort.natsorted(fp_patches)
    
    coords = np.zeros((len(fp_patches), 4), dtype=int)
    
    for i, fp in enumerate(fp_patches):
        
        coords_patch = [int(x) for x in re.findall(r'\d+', os.path.basename(fp))]
        coords[i,:] = np.array(coords_patch)
            
    height_patch = coords[0,1] - coords[0,0]
    width_patch = coords[0,3] - coords[0,2]
    height = np.max(coords[:,1]) + height_patch
    width = np.max(coords[:,2]) + width_patch
    
    whole = np.zeros((height, width), dtype=np.uint16)
    
    for i, fp in enumerate(fp_patches):
        hs, he, ws, we = coords[i,0], coords[i,1], coords[i,2], coords[i,3]
        whole[hs:hs+height_patch, ws:ws+width_patch] = tifffile.imread(fp)
    
    height_trim = np.max(coords[:,1])
    width_trim = np.max(coords[:,3])
    
    whole = whole[:height_trim, :width_trim]
    print(whole.shape)

    tifffile.imwrite(dir_patches+"/all_genes_sum.tif", whole, photometric='minisblack')
   

def main(config):

    """
    Generates transcript expression maps from transcripts.csv.gz, which contains transcript data with locations.
    Example file for Xenium:

        "transcript_id","cell_id","overlaps_nucleus","feature_name","x_location","y_location","z_location","qv"
        281474976710656,565,0,"SEC11C",4.395842,328.66647,12.019493,18.66248
        281474976710657,540,0,"NegControlCodeword_0502",5.074415,236.96484,7.6085105,18.634956
        281474976710658,562,0,"SEC11C",4.702023,322.79715,12.289083,18.66248
        281474976710659,271,0,"DAPK3",4.9066014,581.42865,11.222615,20.821745
        281474976710660,291,0,"TCIM",5.6606994,720.85175,9.265523,18.017488
        281474976710661,297,0,"TCIM",5.899098,748.5928,9.818688,18.017488

    """
    
    dir_dataset = config.data_dir + '/' + config.dataset
    dir_out_maps = dir_dataset + '/' + config.dir_out_maps
    if not os.path.exists(dir_out_maps):
        os.makedirs(dir_out_maps)

    fp_out_filtered = dir_dataset + '/' + config.fp_out_filtered

    # Names to filter out
    fp_transcripts_to_filter = config.data_dir + '/' + config.fp_transcripts_to_filter
    with open(fp_transcripts_to_filter) as file:
        transcripts_to_filter = [line.rstrip() for line in file]

    # Column names in the transcripts csv
    x_col = config.x_col
    y_col = config.y_col
    gene_col = config.gene_col

    if not os.path.exists(fp_out_filtered):
        print('Loading transcripts file')
        fp_transcripts =  dir_dataset + '/' + config.fp_transcripts
        if pathlib.Path(fp_transcripts).suffixes[-1] == '.gz':
            if config.delimiter == "tab":
                df = pd.read_csv(fp_transcripts, sep='\t', compression='gzip')
            else:
                df = pd.read_csv(fp_transcripts, compression='gzip')
        else:
            df = pd.read_csv(fp_transcripts)
        print(df.head())
    
        print('Filtering transcripts')
        if "qv" in df.columns:
            df = df[(df["qv"] >= config.min_qv) &
                    (~df[gene_col].str.startswith(tuple(transcripts_to_filter)))]
        else: 
            df = df[(~df[gene_col].str.startswith(tuple(transcripts_to_filter)))]

        if config.fp_selected_genes is not None:
            with open(dir_dataset + '/' + config.fp_selected_genes) as file:
                selected_genes = [line.rstrip() for line in file]
            df = df[(df[gene_col].isin(selected_genes))]

        # Scale
        print(df[x_col].min(), df[x_col].max(), df[y_col].min(), df[y_col].max())
        df[x_col] = df[x_col].mul(config.scale_ts_x)
        df[y_col] = df[y_col].mul(config.scale_ts_y)
        print(df[x_col].min(), df[x_col].max(), df[y_col].min(), df[y_col].max())

        # Shift
        min_x = df[x_col].min()
        min_y = df[y_col].min()
        if config.shift_to_origin:            
            with pd.option_context('mode.chained_assignment', None):
                df.loc[:, x_col] = df[x_col] - min_x + config.global_shift_x
                df.loc[:, y_col] = df[y_col] - min_y + config.global_shift_y

        size_x = df[x_col].max() + 1
        size_y = df[y_col].max() + 1
        
        # Write transform parameters to file
        fp_affine = os.path.join(dir_dataset, config.fp_affine)
        params = ["scale_ts_x", "scale_ts_y", 
                  "min_x", "min_y", "size_x", "size_y",
                  "global_shift_x", "global_shift_y", "origin"]
        vals = [config.scale_ts_x, config.scale_ts_y, 
                min_x, min_y,
                size_x, size_y,
                config.global_shift_x, config.global_shift_y,
                config.shift_to_origin]
        with open(fp_affine, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(params,vals))

        # Delete entries with negative coordinates
        df = df[df[x_col] >= 0]
        df = df[df[y_col] >= 0]
    
        df.reset_index(inplace=True, drop=True)
        print('Finished filtering')
        print('Saving csv...')

        df.to_csv(fp_out_filtered)
    else:
        print('Loading filtered transcripts')
        df = pd.read_csv(fp_out_filtered, index_col=0)
    
    # Round locations and convert to integer
    df[x_col] = df[x_col].round().astype(int)
    df[y_col] = df[y_col].round().astype(int)

    print(df.head)
    print(df.shape)

    # Save list of gene names
    gene_names = df[gene_col].unique()
    print('%d unique genes' %len(gene_names))
    gene_names = natsort.natsorted(gene_names)
    with open(dir_dataset+'/'+config.fp_out_gene_names, 'w') as f:
        for line in gene_names:
            f.write(f"{line}\n")

    # Dimensions
    total_height_t = int(np.ceil(df[y_col].max())) + 1
    total_width_t = int(np.ceil(df[x_col].max())) + 1

    fp_nuclei = os.path.join(dir_dataset, config.fp_nuclei)
    if os.path.exists(fp_nuclei):
        nuclei_img = tifffile.imread(fp_nuclei)
        nuclei_h = nuclei_img.shape[0]
        nuclei_w = nuclei_img.shape[1]
        if total_height_t <= nuclei_h and total_width_t <= nuclei_w:
            total_height = nuclei_h
            total_width = nuclei_w
        else:
            sys.exit(f"Dimensions of transcript map [{total_height_t},{total_width_t}] exceeds those of nuclei image [{nuclei_h},{nuclei_w}]. Check scale_ts_x and scale_ts_y values. Then consider specifying --global_shift_x and --global_shift_y, or padding nuclei")
    else:
        warnings.warn("Computing dimensions from transcript locations - check dimensions are the same as nuclei image. Unless cropping DAPI to size of transcript map, it is highly advised to provide nuclei file name via --fp_nuclei to ensure dimensions are identical")
        total_height = total_height_t
        total_width = total_width_t

    print(f"Total height {total_height}, width {total_width}")

    # Start and end coordinates of patches
    h_coords, img_height = get_patches_coords(total_height, config.max_height)
    w_coords, img_width = get_patches_coords(total_width, config.max_width)
    hw_coords = [(hs, he, ws, we) for (hs, he) in h_coords for (ws, we) in w_coords]
    
    print('Converting to maps')

    n_processes = get_n_processes(config.n_processes)
    gene_names_chunks = np.array_split(gene_names, n_processes)

    for (hs, he, ws, we) in tqdm(hw_coords):
        print("Patch:", (hs, he, ws, we))

        df_patch = df[(df[x_col].between(ws, we-1)) & (df[y_col].between(hs, he-1))]
        
        with pd.option_context('mode.chained_assignment', None):
            df_patch.loc[:, x_col] = df_patch[x_col] - ws
            df_patch.loc[:, y_col] = df_patch[y_col] - hs

        df_patch.reset_index(inplace=True, drop=True)

        processes = []

        for gene_chunk in gene_names_chunks:
            p = mp.Process(target=process_gene_chunk, args=(gene_chunk, df_patch, 
                                                            img_height, img_width,
                                                            dir_out_maps, hs, ws,
                                                            gene_col, x_col, y_col,
                                                            config.counts_col))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        
        # Combine channel-wise
        map_all_genes = np.zeros((img_height, img_width, len(gene_names)), dtype=np.uint8)
        
        for i_fe, fe in enumerate(tqdm(gene_names)):
            fp_fe_map = f"{dir_out_maps}/{fe}_{hs}_{ws}.tif"
            map_all_genes[:,:,i_fe] = tifffile.imread(fp_fe_map)
            os.remove(fp_fe_map)

        # Sum across all markers
        fp_out_map_sum = f"all_genes_sum_{hs}_{he}_{ws}_{we}.tif"
        tifffile.imwrite(dir_out_maps+'/'+fp_out_map_sum, np.sum(map_all_genes, -1).astype(np.uint16), photometric='minisblack')

        # Save to hdf5
        fp_out_map = f"all_genes_{hs}_{he}_{ws}_{we}.hdf5"
        h = h5py.File(dir_out_maps+'/'+fp_out_map, 'w')
        dset = h.create_dataset('data', data=map_all_genes, dtype=np.uint8)
        
    print('Saved all maps')
    
    stitch_patches(dir_out_maps, '/all_genes_sum_*.tif')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../../data/', type=str)
    parser.add_argument('--dataset', default='dataset_merscope_melanoma2', type=str)

    parser.add_argument('--n_processes', default=None, type=int)

    parser.add_argument('--fp_transcripts', default='HumanMelanomaPatient2_detected_transcripts.csv', type=str)
    parser.add_argument('--min_qv', default=20, type=int)
    parser.add_argument('--fp_transcripts_to_filter', default='transcripts_to_filter.txt', type=str)

    parser.add_argument('--dir_out_maps', default='expr_maps', type=str)
    parser.add_argument('--fp_out_filtered', default='transcripts_processed.csv', type=str)
    parser.add_argument('--fp_out_gene_names', default='all_gene_names.txt', type=str)

    # Size of the images - divide into sections if too large
    parser.add_argument('--scale_ts_x', default=1.0, type=float, help="conversion between transcript location values and target pixel resolution along width")
    parser.add_argument('--scale_ts_y', default=1.0, type=float, help="conversion between transcript location values and target pixel resolution along height")
    parser.add_argument('--max_height', default=3500, type=int, help="Height of patches")
    parser.add_argument('--max_width', default=4000, type=int, help="Width of patches")

    parser.add_argument('--fp_nuclei', default='nuclei.tif', type=str)
    parser.add_argument('--fp_affine', default='affine.csv', type=str)

    # Shift to origin, making min(x) and min(y) (0,0)
    parser.add_argument('--shift_to_origin', action='store_true')
    parser.set_defaults(shift_to_origin=False)

    # Additional adjustment to align to DAPI
    parser.add_argument('--global_shift_x', default=0, type=int)
    parser.add_argument('--global_shift_y', default=0, type=int)

    # Names of columns
    parser.add_argument('--x_col', default='global_x', type=str)
    parser.add_argument('--y_col', default='global_y', type=str)
    parser.add_argument('--gene_col', default='gene', type=str)

    # BGI Stereo-seq
    parser.add_argument('--fp_selected_genes', default=None, type=str) # selected_genes.txt
    parser.add_argument('--counts_col', default=None, type=str) # MIDCounts
    parser.add_argument('--delimiter', default=None, type=str)


    config = parser.parse_args()
    main(config)
