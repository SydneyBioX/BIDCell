import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np
import tifffile
import pandas as pd 
import glob
import natsort
import re
import bisect

from dataio.dataset_input import DataProcessing
from model.model import UNet3Plus as Network
from utils.utils import *

def main(config):

    json_opts = json_file_to_pyobj(config.config_file)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create experiment directories
    make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir)
    experiment_path = 'experiments' + '/' + timestamp
    model_dir = experiment_path + '/' + json_opts.experiment_dirs.model_dir
    test_output_dir = experiment_path + '/' + json_opts.experiment_dirs.test_output_dir
    make_dir(test_output_dir)

    # Set up the model
    logging.info("Initialising model")
   
    atlas_exprs = pd.read_csv(json_opts.data_sources.atlas_fp, index_col=0)
    n_genes = atlas_exprs.shape[1] - 3
    print('Number of genes: %d' %n_genes)

    model = Network(n_channels=n_genes)
    model = model.to(device)

    # Get list of model files
    if config.test_epoch < 0:
        saved_model_paths, _ = get_files_list(model_dir, ['.pth'])
        saved_model_paths = sorted_alphanumeric(saved_model_paths)
        saved_model_names = [(os.path.basename(x)).split('.')[0] for x in saved_model_paths]
        saved_model_epochs = [x.split('_')[1] for x in saved_model_names]
        saved_model_steps = [x.split('_')[-1] for x in saved_model_names]
        if config.test_epoch == None:
            saved_model_epochs = np.array(saved_model_epochs, dtype='int')
            saved_model_steps = np.array(saved_model_steps, dtype='int')
        elif config.test_epoch == -1:
            saved_model_epochs = np.array(saved_model_epochs[-1], dtype='int')
            saved_model_epochs = [saved_model_epochs]
            saved_model_steps = np.array(saved_model_steps[-1], dtype='int')
            saved_model_steps = [saved_model_steps]
    else:
        saved_model_epochs = [config.test_epoch]
        saved_model_steps = [config.test_step]

    shifts = [0, int(json_opts.data_params.patch_size/2)]

    for shift_patches in shifts:

        # Dataloader
        logging.info("Preparing data")
        test_dataset = DataProcessing(json_opts.data_sources,
                                    json_opts.data_params,
                                    isTraining=False,
                                    shift_patches=shift_patches)
        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=1, 
                                shuffle=False, num_workers=0)

        n_test_examples = len(test_loader)
        logging.info("Total number of testing examples: %d" %n_test_examples)

        logging.info("Begin testing")
        
        for epoch_idx, (test_epoch, test_step) in enumerate(zip(saved_model_epochs, saved_model_steps)):
            current_dir = test_output_dir + '/' + 'epoch_' + str(test_epoch) + '_step_' + str(test_step)
            make_dir(current_dir)

            # Restore model
            load_path = model_dir + "/epoch_%d_step_%d.pth" %(test_epoch, test_step)
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            assert(epoch == test_epoch)
            print("Testing " + load_path)

            model = model.eval()

            for batch_idx, (batch_x313, batch_n, batch_sa, batch_pos, batch_neg, coords_h1, coords_w1, nucl_aug, expr_aug_sum, whole_h, whole_w, expr_fp) in enumerate(test_loader):
            
                if batch_idx == 0:
                    whole_seg = np.zeros((whole_h, whole_w), dtype=np.uint32)
                    
                # Permute channels axis to batch axis
                batch_x313 = batch_x313[0,:,:,:,:].permute(3,2,0,1)
                batch_sa = batch_sa.permute(3,0,1,2)
                batch_n = batch_n.permute(3,0,1,2)
                
                if batch_x313.shape[0] == 0:
                    seg_patch = np.zeros((json_opts.data_params.patch_size, json_opts.data_params.patch_size), dtype=np.uint32)

                else:
                    # Transfer to GPU
                    batch_x313 = batch_x313.to(device)
                    batch_sa = batch_sa.to(device)
                    batch_n = batch_n.to(device)

                    # Forward pass
                    seg_pred = model(batch_x313)

                    coords_h1 = coords_h1.detach().cpu().squeeze().numpy()
                    coords_w1 = coords_w1.detach().cpu().squeeze().numpy()
                    sample_seg = seg_pred.detach().cpu().numpy()
                    sample_n = nucl_aug.detach().cpu().numpy()
                    sample_sa = batch_sa.detach().cpu().numpy()
                    sample_expr = expr_aug_sum.detach().cpu().numpy()
                    patch_fp = current_dir + \
                            "/%d_%d.png" %(coords_h1, coords_w1)
                            
                    if (batch_idx % json_opts.save_freqs.sample_freq) == 0: 
                        save_fig_outputs(sample_seg, sample_n, sample_sa, sample_expr, patch_fp)
                    
                    seg_patch = get_seg_mask(sample_seg, sample_n)
                
                # seg_patch_fp = current_dir + '/' + "%d_%d.tif" %(coords_h1, coords_w1)
                # tifffile.imwrite(seg_patch_fp, seg_patch.astype(np.uint32), photometric='minisblack')
                    
                whole_seg[coords_h1:coords_h1+json_opts.data_params.patch_size, 
                        coords_w1:coords_w1+json_opts.data_params.patch_size] = seg_patch.copy()
                    
            seg_fp = test_output_dir + '/' + "epoch_%d_step_%d_seg_shift%d.tif" %(test_epoch, test_step, shift_patches)
            
            tifffile.imwrite(seg_fp, whole_seg.astype(np.uint32), photometric='minisblack')

    logging.info("Testing finished")

    return test_output_dir


def gap_coords(coords, patcsize):
    """If gap larger than patcsize -> remove all corresponding locations"""
    starts_diff = np.diff(coords)
    gap_idx = np.where(starts_diff > patcsize)[0] + 1
    # print(gap_idx.shape)
    gap_end = [coords[x] for x in gap_idx]
    gap_start = [coords[bisect.bisect(coords, x-1)-1] for x in gap_end]
    # print(gap_start, gap_end)
    gap = []
    for gs, ge in zip(gap_start, gap_end):
        gap.extend(list(range(gs, ge)))
    # print(len(gap))
    return gap


def fill_grid(config, dir_id):
    """
    Combine predictions from unshifted and shifted patches to remove 
    border effects
    """
    
    print("Combining predictions")

    json_opts = json_file_to_pyobj(config.config_file)
    patch_size = json_opts.data_params.patch_size
    shift = int(patch_size/2)
    
    expr_fp = json_opts.data_sources.expr_fp + str(json_opts.data_params.patch_size) + \
              'x' + str(json_opts.data_params.patch_size) + '_shift_' + str(shift)
    expr_fp_ext = json_opts.data_sources.expr_fp_ext
    
    pred_fp = '%s/epoch_%d_step_%d_seg_shift0.tif' \
                   %(dir_id, config.test_epoch, config.test_step)
    pred_fp_sf = '%s/epoch_%d_step_%d_seg_shift%d.tif' \
                   %(dir_id, config.test_epoch, config.test_step, shift)

    output_fp = dir_id + '/' + os.path.basename(pred_fp).replace('_seg_shift0','')

    pred = tifffile.imread(pred_fp)
    pred_sf = tifffile.imread(pred_fp_sf)

    fp_patches_sf = glob.glob(expr_fp + '/*' + expr_fp_ext)
    fp_patches_sf = natsort.natsorted(fp_patches_sf)

    coords_patches = [re.findall(r'\d+', os.path.basename(x)) for x in fp_patches_sf]
    coords_h1 = [int(x[0]) for x in coords_patches]
    coords_w1 = [int(x[1]) for x in coords_patches]
    
    # Fill along grid
    h_starts_wide = []
    w_starts_wide = []

    # Middle section of shifted patches
    for i in range(int(patch_size*0.35), int(patch_size*0.65)):
        h_starts_wide.extend([x+i for x in coords_h1])
        w_starts_wide.extend([x+i for x in coords_w1])

    # Gap larger than patch_size -> remove all corresponding locations
    h_gap = gap_coords(coords_h1, patch_size)
    w_gap = gap_coords(coords_w1, patch_size)
            
    fill = np.zeros(pred.shape)
    fill[h_starts_wide,:] = 1
    fill[:,w_starts_wide] = 1

    # Gaps
    fill[h_gap,:] = 0
    fill[:,w_gap] = 0

    tifffile.imwrite(dir_id + '/' +'fill.tif', fill.astype(np.uint16), photometric='minisblack')    

    result = np.zeros(pred.shape, dtype=np.uint32)
    result = np.where(fill > 0, pred_sf, pred)

    tifffile.imwrite(output_fp, result.astype(np.uint32), photometric='minisblack')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config.json', type=str,
                        help='config file path')
    parser.add_argument('--test_epoch', default=1, type=int,
                        help='test model from this epoch, -1 for last, None for all')
    parser.add_argument('--test_step', default=4000, type=int,
                        help='test model from this step')

    config = parser.parse_args()

    test_output_dir = main(config)
    
    # test_output_dir = 'experiments/2023_July_06_09_41_41/test_output'
    
    fill_grid(config, test_output_dir)
