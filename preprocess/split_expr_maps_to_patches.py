import h5py 
import numpy as np
import os
from tqdm import tqdm 
import argparse

def main(config):
    """
    Divide transcriptomic maps of all genes created by generate_expr_maps.py 
    into patches and save as hdf5
    
    """

    patch_size = config.patch_size
    shift = [0, int(patch_size/2)]

    fp_maps = config.fp_maps
    h5f = h5py.File(fp_maps, 'r')
    xenium = h5f['data'][:]
    h5f.close()
    print('Loaded gene expr maps from %s' %fp_maps)

    print(xenium.shape)

    for shift_patches in shift:
        print('Shift by %d' %shift_patches)
    
        # Get coordinates of non-overlapping patches
        if shift_patches == 0:
            h_starts = list(np.arange(0, xenium.shape[0]-patch_size, patch_size))
            w_starts = list(np.arange(0, xenium.shape[1]-patch_size, patch_size))
            
            # Include remainder patches on 
            h_starts.append(xenium.shape[0]-patch_size)
            w_starts.append(xenium.shape[1]-patch_size)
            
            dir_output = '../data/patches_%dx%d/' %(patch_size, patch_size)
            
        else:
            h_starts = list(np.arange(shift_patches, xenium.shape[0]-patch_size, patch_size))
            w_starts = list(np.arange(shift_patches, xenium.shape[1]-patch_size, patch_size))    

            dir_output = '../data/patches_%dx%d_shift_%d/' %(patch_size, patch_size, shift_patches)

        coords_starts = [(x, y) for x in h_starts for y in w_starts]

        if not os.path.exists(dir_output):
            os.makedirs(dir_output)
        
        # Get patches and save 
        for (h,w) in tqdm(coords_starts):
            patch = xenium[h:h+patch_size, w:w+patch_size, :]

            fp_output = dir_output + str(h) + 'x' + str(w) + '.hdf5'

            h = h5py.File(fp_output, 'w')
            dset = h.create_dataset('data', data=patch, dtype=np.uint8)
            h.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fp_maps', default='../data/expr_maps/all_genes.hdf5', type=str)   
    parser.add_argument('--patch_size', default=48, type=int)

    config = parser.parse_args()
    main(config)