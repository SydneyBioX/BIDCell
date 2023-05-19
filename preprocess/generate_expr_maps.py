import tifffile
import pandas as pd
import numpy as np
import h5py
import os
import natsort
import argparse
from tqdm import tqdm 
import multiprocessing as mp

def process_gene_chunk(gene_chunk, df, map_height, map_width, dir_output):
    for i_fe, fe in enumerate(gene_chunk):
        df_fe = df.loc[df['feature_name'] == fe]
        map_fe = np.zeros((map_height, map_width))

        for idx in df_fe.index:
            idx_x = np.round(df.iloc[idx]['x_location']).astype(int)
            idx_y = np.round(df.iloc[idx]['y_location']).astype(int)
            idx_qv = df.iloc[idx]['qv']

            map_fe[idx_y, idx_x] += 1
        
        tifffile.imwrite(dir_output+'/'+fe+'.tif', map_fe.astype(np.uint8), photometric='minisblack')


def main(config):

    """
    Generates transcript expression maps from transcripts.csv.gz, which contains transcript data with locations:

        "transcript_id","cell_id","overlaps_nucleus","feature_name","x_location","y_location","z_location","qv"
        281474976710656,565,0,"SEC11C",4.395842,328.66647,12.019493,18.66248
        281474976710657,540,0,"NegControlCodeword_0502",5.074415,236.96484,7.6085105,18.634956
        281474976710658,562,0,"SEC11C",4.702023,322.79715,12.289083,18.66248
        281474976710659,271,0,"DAPK3",4.9066014,581.42865,11.222615,20.821745
        281474976710660,291,0,"TCIM",5.6606994,720.85175,9.265523,18.017488
        281474976710661,297,0,"TCIM",5.899098,748.5928,9.818688,18.017488

    """

    dir_output = config.dir_output
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    if not os.path.exists(config.fp_out_filtered):
        print('Loading transcripts file')
        df = pd.read_csv(config.fp_transcripts, compression='gzip')
        print(df.head())
    
        print('Filtering transcripts')
        df = df[(df["qv"] >= config.min_qv) &
                (~df["feature_name"].str.startswith("NegControlProbe_")) &
                (~df["feature_name"].str.startswith("antisense_")) &
                (~df["feature_name"].str.startswith("NegControlCodeword_")) &
                (~df["feature_name"].str.startswith("BLANK_"))]
        
        df.reset_index(inplace=True, drop=True)
        print('Finished filtering')
        print('...')

        df.to_csv(config.fp_out_filtered)
    else:
        print('Loading filtered transcripts')
        df = pd.read_csv(config.fp_out_filtered, index_col=0)
    
    print(df.head())

    gene_names = df['feature_name'].unique()
    print('%d unique genes' %len(gene_names))

    gene_names = natsort.natsorted(gene_names)

    with open(dir_output+'/'+config.fp_out_gene_names, 'w') as f:
        for line in gene_names:
            f.write(f"{line}\n")

    map_width = int(np.ceil(df['x_location'].max())) + 1
    map_height = int(np.ceil(df['y_location'].max())) + 1
    print("W: %d, H: %d" %(map_width, map_height))

    print('Converting to maps')

    if config.n_processes == None:
        n_processes = mp.cpu_count()
    else:
        n_processes = config.n_processes if config.n_processes <= mp.cpu_count() else mp.cpu_count()
    
    gene_names_chunks = np.array_split(gene_names, n_processes)
    processes = []

    for gene_chunk in gene_names_chunks:
        p = mp.Process(target=process_gene_chunk, args=(gene_chunk, df, 
                                                        map_height, map_width,
                                                        config.dir_output))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    # Combine channel-wise
    map_all_genes = np.zeros((map_height, map_width, len(gene_names)), dtype=np.uint8)
    
    for i_fe, fe in enumerate(tqdm(gene_names)):
        map_all_genes[:,:,i_fe] = tifffile.imread(dir_output+'/'+fe+'.tif')

    # Sum across all markers
    tifffile.imwrite(dir_output+'/'+config.fp_out_map_sum, np.sum(map_all_genes, -1).astype(np.uint16), photometric='minisblack')

    # Save to hdf5
    h = h5py.File(dir_output+'/'+config.fp_out_map, 'w')
    dset = h.create_dataset('data', data=map_all_genes, dtype=np.uint8)
    
    print('Saved maps')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fp_transcripts', default='transcripts.csv.gz', type=str)
    parser.add_argument('--min_qv', default=20, type=int)
    parser.add_argument('--n_processes', default=None, type=int)

    parser.add_argument('--dir_output', default='../data/expr_maps', type=str)
    parser.add_argument('--fp_out_filtered', default='../data/transcripts_filtered.csv', type=str)
    parser.add_argument('--fp_out_gene_names', default='all_gene_names.txt', type=str)
    parser.add_argument('--fp_out_map', default='all_genes.hdf5', type=str)
    parser.add_argument('--fp_out_map_sum', default='all_genes_sum.tif', type=str)

    config = parser.parse_args()
    main(config)
