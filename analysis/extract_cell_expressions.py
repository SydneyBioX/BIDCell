import numpy as np 
import h5py
import tifffile
from tqdm import tqdm
import argparse
import os
import cv2 
from scipy.stats import spearmanr
import pandas as pd
import multiprocessing as mp
import glob

def process_chunk(chunk, output_dir, cell_ids_unique, col_names, seg_map):
    df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
    df_out["cell_id"] = cell_ids_unique.copy()

    chunk_id = chunk.index[0]

    for index_row, row in chunk.iterrows():
        gene = row["feature_name"]
        w_loc = row["x_location"]
        h_loc = row["y_location"]
        
        seg_val = seg_map[h_loc, w_loc]
        if seg_val > 0:
            df_out.loc[seg_val, gene] += 1

    df_out.to_csv(output_dir+'/'+'chunk_%d.csv' %chunk_id)


def process_chunk_meta(matrix, fp_output, seg_map_mi, col_names_coords):

    chunk_id = matrix[0,0]
    output = np.zeros((matrix.shape[0], len(col_names_coords)))
    output[:,0] = matrix[:,0].copy()
    output[:,4:] = matrix[:,1:].copy()

    for cur_i, cell_id in enumerate(output[:,0]):
        if cell_id > 0:
            try:
                # cell_centroid_x and cell_centroid_y
                coords = np.where(seg_map_mi == cell_id)
                x_points = coords[1]
                y_points = coords[0]
                centroid_x = sum(x_points) / len(x_points)
                centroid_y = sum(y_points) / len(y_points)
                output[cur_i,1] = centroid_x.copy()
                output[cur_i,2] = centroid_y.copy()

                # cell_size
                output[cur_i,3] = len(coords[0])
            except:
                output[cur_i,1] = -1
                output[cur_i,2] = -1
                output[cur_i,3] = -1

    # Save as csv
    df_split = pd.DataFrame(output, index=list(range(output.shape[0])), columns=col_names_coords)
    df_split.to_csv(fp_output + '%d.csv' %chunk_id, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fp_seg', default='../BIDCell_model/experiments/2023_April_18_19_31_46/test_output/epoch_1_step_4000_connected.tif', type=str,
                        help='segmentation')
    parser.add_argument('--output_dir', default='cell_gene_matrices/2023_April_18_19_31_46', type=str,
                        help='output directory')
    parser.add_argument('--fp_output_expr', default='cell_expr.csv', type=str,
                        help='output file')
    parser.add_argument('--fp_output_full', default='cell_outputs_', type=str,
                        help='output file')
    parser.add_argument('--fp_transcripts', default='../preprocess/transcripts.csv.gz', type=str,
                        help='downloaded transcripts data')
    parser.add_argument('--fp_transcripts_for_matrix', default='transcripts_for_matrix.csv', type=str,
                        help='filtered and xy-scaled transcripts data')
    parser.add_argument('--fp_gene_names', default='../data/expr_maps/all_gene_names.txt', type=str,
                        help='txt file containing list of gene names')                    
    parser.add_argument('--scaling_factor', default=0.2125, type=float,
                        help='conversion between pixel size and microns')  
    parser.add_argument('--min_qv', default=20, type=int,
                        help='min acceptable qv')  
    parser.add_argument('--n_processes', default=None, type=int)

    config = parser.parse_args()
   

    fp_seg = config.fp_seg

    output_dir = config.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_map_mi = tifffile.imread(fp_seg)
    height = seg_map_mi.shape[0]
    width = seg_map_mi.shape[1]

    cell_ids_unique = np.unique(seg_map_mi.reshape(-1))
    cell_ids_unique = cell_ids_unique[1:]
    n_cells = len(cell_ids_unique)
    print("Number of cells " + str(n_cells))

    with open(config.fp_gene_names) as file:
        gene_names = [line.rstrip() for line in file]

    col_names = ["cell_id"] + gene_names
    
    # Divide the dataframe into chunks for multiprocessing 
    if config.n_processes == None:
        n_processes = mp.cpu_count()
    else:
        n_processes = config.n_processes if config.n_processes <= mp.cpu_count() else mp.cpu_count()
            
    if not os.path.exists(output_dir + '/' + config.fp_output_expr):
        # Rescale to pixel size
        height_pix = np.round(height/config.scaling_factor).astype(int)
        width_pix = np.round(width/config.scaling_factor).astype(int)
        seg_map = cv2.resize(seg_map_mi.astype(np.int32), (width_pix, height_pix), interpolation = cv2.INTER_NEAREST)
        print('Segmentation map size: ', seg_map.shape)
        tifffile.imwrite(output_dir+'/rescaled.tif', seg_map.astype(np.uint32), photometric='minisblack')

        if not os.path.exists(config.fp_transcripts_for_matrix):
            # Get transcripts data
            df_expr = pd.read_csv(config.fp_transcripts, compression='gzip')
            print('Filtering transcripts')

            df_expr = df_expr[(df_expr["qv"] >= config.min_qv) &
                                (~df_expr["feature_name"].str.startswith("NegControlProbe_")) &
                                (~df_expr["feature_name"].str.startswith("antisense_")) &
                                (~df_expr["feature_name"].str.startswith("NegControlCodeword_")) &
                                (~df_expr["feature_name"].str.startswith("BLANK_"))]

            df_expr = df_expr.assign(x_location = df_expr["x_location"]/config.scaling_factor)
            df_expr = df_expr.assign(y_location = df_expr["y_location"]/config.scaling_factor)
            
            df_expr["x_location"] = df_expr["x_location"].round(0).astype(int)
            df_expr["y_location"] = df_expr["y_location"].round(0).astype(int)
            
            df_expr = df_expr[(df_expr["x_location"] >= 0) &
                                (df_expr["x_location"] < width_pix) &
                                (df_expr["y_location"] >= 0) &
                                (df_expr["y_location"] < height_pix)]
            
            df_expr.to_csv(config.fp_transcripts_for_matrix, index=None)
            print('Filtered transcripts')

        else:
            df_expr = pd.read_csv(config.fp_transcripts_for_matrix)
            print('Read filtered transcripts')

        df_expr.reset_index(drop=True, inplace=True)
        
        df_expr_splits = np.array_split(df_expr, n_processes)
        print('Number of splits for multiprocessing: %d' %len(df_expr_splits))
        print('Extracting cell expressions')
        processes = []

        for chunk in df_expr_splits:
            p = mp.Process(target=process_chunk, args=(chunk, output_dir, 
                                                        cell_ids_unique, col_names,
                                                        seg_map))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print('Obtained cell-gene matrix chunks')

        df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
        df_out["cell_id"] = cell_ids_unique.copy()

        print('Combining cell-gene matrix chunks')

        fp_chunks = glob.glob(output_dir + '/chunk_*.csv')
        for fpc in fp_chunks:
            df_i = pd.read_csv(fpc, index_col=0)
            df_out.iloc[:,1:] = df_out.iloc[:,1:].add(df_i.iloc[:,1:])
            
        print('Obtained cell-gene matrix')

        df_out.to_csv(output_dir + '/' + config.fp_output_expr)
        
        # Clean up
        for fpc in fp_chunks:
            os.remove(fpc)
        os.remove(output_dir+'/rescaled.tif')

        del seg_map 
    
    else:
        df_out = pd.read_csv(output_dir + '/' + config.fp_output_expr, index_col=0)
   
    print('Computing locations and sizes')

    matrix_all = df_out.to_numpy().astype(np.float32)
    matrix_all_splits = np.array_split(matrix_all, n_processes)
    processes = []

    fp_output = output_dir + '/' + config.fp_output_full 
    col_names_coords = ["cell_id", "cell_centroid_x", "cell_centroid_y", "cell_size"] + gene_names

    for chunk in matrix_all_splits:
        p = mp.Process(target=process_chunk_meta, args=(chunk, fp_output, 
                                                        seg_map_mi, col_names_coords))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print('Done')