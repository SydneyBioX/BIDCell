import numpy as np 
import matplotlib.pyplot as plt 
import tifffile 
import os
import argparse
import re
import random
import cv2
from scipy import ndimage as ndi
import glob 
import multiprocessing as mp

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
    
   
def get_exp_dir(config):
    if config.dir_id == 'last':
        folders = next(os.walk('experiments'))[1]
        folders = sorted_alphanumeric(folders)
        folder_last = folders[-1]
        dir_id = folder_last.replace('\\','/')
    else:
        dir_id = config.dir_id

    return dir_id
    

def postprocess_connect(img, nuclei):
    cell_ids = np.unique(img)
    cell_ids = cell_ids[1:]

    # Random order
    random.shuffle(cell_ids)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    # Touch diagonally = same object
    s = ndi.generate_binary_structure(2,2)

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
            for i_part in range(1,len(counts)):
                if i_part > 1:
                    no_overlap = True
                largest = unique[np.argmax(counts[i_part:])+i_part]
                connected_mask = np.where(unique_ids == largest, 1, 0)
                # Break if current largest region overlaps nucleus
                if np.sum(connected_mask*nucleus_mask) > 0.5:
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

    # Random order
    random.shuffle(cell_ids)

    # Touch diagonally = same object
    s = ndi.generate_binary_structure(2,2)

    final = np.zeros(img.shape, dtype=np.uint32)

    for i in cell_ids:
        i_mask = np.where(img == i, 1, 0).astype(np.uint8)

        nucleus_mask = np.where(nuclei == i, 1, 0).astype(np.uint8)

        # Number of blobs belonging to cell
        unique_ids, num_blobs = ndi.label(i_mask, structure=s)
        if num_blobs > 1:
            # Keep the blob with max overlap to nucleus 
            amount_overlap = np.zeros(num_blobs)

            for i_blob in range(1,num_blobs+1):
                blob = np.where(unique_ids == i_blob, 1, 0)
                amount_overlap[i_blob-1] = np.sum(blob*nucleus_mask)
            blob_keep = np.argmax(amount_overlap) + 1
            
            final_mask = np.where(unique_ids == blob_keep, 1, 0)

        else:
            blob_size = np.count_nonzero(i_mask)
            if blob_size > 2:
                final_mask = i_mask.copy()
            else:
                final_mask = i_mask*0

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

        output_fp = output_dir + '%d_%d.tif' %(coords_x1, coords_y1)

        # print('Filling holes')
        filled = postprocess_connect(img, nuclei)
        
        # print('Removing islands')
        final = remove_islands(filled, nuclei)

        tifffile.imwrite(output_fp, final.astype(np.uint32), photometric='minisblack')

        cell_ids = np.unique(final)[1:]

        # Visualise cells with random colours
        n_cells_ids = len(cell_ids)
        cell_ids_rand = np.arange(1, n_cells_ids+1)
        random.shuffle(cell_ids_rand)

        keep_mask = np.isin(nuclei, cell_ids)
        nuclei = np.where(keep_mask==True, nuclei, 0)
        keep_mask = np.isin(img, cell_ids)
        img = np.where(keep_mask==True, nuclei, 0)

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
        ax[0].set_title('Nuclei')
        ax[1].imshow(img_mapped, cmap=plt.cm.nipy_spectral)
        ax[1].set_title('Original')
        ax[2].imshow(final_mapped, cmap=plt.cm.nipy_spectral)
        ax[2].set_title('Processed')
        for a in ax:
            a.set_axis_off()
        fig.tight_layout()
        # plt.show()        

        fig.savefig(output_fp.replace('tif','png'), dpi=300)
        plt.close(fig)



def combine(config, dir_id):
    """
    Combine the patches previously output by the connect function
    """

    fp_dir = dir_id + 'epoch_%d_step_%d_connected' %(config.epoch, config.step)
    fp_unconnected = dir_id + 'epoch_%d_step_%d.tif' %(config.epoch, config.step)
    fp_output_seg = fp_dir + '.tif'

    dl_pred = tifffile.imread(fp_unconnected)
    height = dl_pred.shape[0]
    width = dl_pred.shape[1]

    seg_final = np.zeros((height, width), dtype=np.uint32)

    fp_seg = glob.glob(fp_dir+'/*.tif', recursive=True)

    sample = tifffile.imread(fp_seg[0])
    patch_h = sample.shape[0]
    patch_w = sample.shape[1]

    for fp in fp_seg:
        patch = tifffile.imread(fp)
        fp_coords = os.path.basename(fp).split('.')[0]
        fp_x = int(fp_coords.split('_')[0])
        fp_y = int(fp_coords.split('_')[1])
        seg_final[fp_x:fp_x+patch_h, fp_y:fp_y+patch_w] = patch[:]

    tifffile.imwrite(fp_output_seg, seg_final.astype(np.uint32), photometric='minisblack')
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_id', default='last', type=str)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--step', default=4000, type=int)
    parser.add_argument('--nucleus_fp', default='../data/nuclei.tif', type=str)
    parser.add_argument('--patch_size', default=1024, type=int)

    config = parser.parse_args()
    
    dir_id = "experiments/" + get_exp_dir(config) + '/test_output/'

    pred_fp = dir_id + 'epoch_%d_step_%d.tif' %(config.epoch, config.step)
    output_dir = dir_id + 'epoch_%d_step_%d_connected/' %(config.epoch, config.step)

    nucleus_fp = config.nucleus_fp
    nuclei_img = tifffile.imread(nucleus_fp)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_whole = tifffile.imread(pred_fp)

    patch_size = config.patch_size
    h_starts = list(np.arange(0, img_whole.shape[0]-patch_size, patch_size))
    w_starts = list(np.arange(0, img_whole.shape[1]-patch_size, patch_size))
    h_starts.append(img_whole.shape[0]-patch_size)
    w_starts.append(img_whole.shape[1]-patch_size)
    coords_starts = [(x, y) for x in h_starts for y in w_starts]
    print('%d patches available' %len(coords_starts))

    num_processes = mp.cpu_count()
    
    coords_splits = np.array_split(coords_starts, num_processes)
    print('Num multiprocessing splits: %d' %len(coords_splits))
    processes = []

    for chunk in coords_splits:
        p = mp.Process(target=process_chunk, args=(chunk, patch_size, img_whole, nuclei_img, output_dir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


    combine(config, dir_id)