import tifffile
import numpy as np
import argparse
import os
import natsort
import glob
import re
from PIL import Image
import sys 

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
    num_hash = pattern.count('#')
    no_hash = pattern.replace('#','')

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
            dapi = dapi[channel_dapi,:,:]
        else:
            dapi = dapi[:,:,channel_dapi]

    return dapi


def main(config):
    dir_dataset = os.path.join(config.data_dir, config.dataset)
    
    if config.dir_dapi == None:
        dir_dapi = dir_dataset
    else:
        dir_dapi = os.path.join(dir_dataset, config.dir_dapi)

    ext_pat = ''.join('[%s%s]' % (e.lower(), e.upper()) for e in config.ext_dapi)
    fp_dapi_list = glob.glob(os.path.join(dir_dapi, "*." + ext_pat))
    fp_dapi_list = natsort.natsorted(fp_dapi_list)

    sample = tifffile.imread(fp_dapi_list[0])
    fov_shape = sample.shape
    
    # Error if multi-channel image and channel is not specified 
    if len(fov_shape) > 2:
        if config.channel_first:
            print(f"Channel axis first, DAPI channel {config.channel_dapi}")
            fov_h = sample.shape[1]
            fov_w = sample.shape[2]
        else:
            print(f"Channel axis last, DAPI channel {config.channel_dapi}")
            fov_h = sample.shape[0]
            fov_w = sample.shape[1]

    fov_dtype = sample.dtype

    # Error if patterns in file path names not found
    found_f = [check_pattern(s, config.pattern_f) for s in fp_dapi_list]
    check_images_meet_criteria(fp_dapi_list, found_f, "FOV string pattern not found in")
    
    if config.pattern_z != None:
        found_z = [check_pattern(s, config.pattern_z) for s in fp_dapi_list]
        check_images_meet_criteria(fp_dapi_list, found_z, "Z slice string pattern not found in")

    # Check shape the same as sample
    match_shape = [check_shape_imgs(s, fov_h, fov_w) for s in fp_dapi_list]
    check_images_meet_criteria(fp_dapi_list, match_shape, "Different image shape for")

    # Locations of each FOV in the whole image
    n_fov = config.n_fov
    if config.row_major:
        order = np.arange(config.n_fov_h*config.n_fov_w).reshape((config.n_fov_h, config.n_fov_w))
    else:
        order = np.arange(config.n_fov_h*config.n_fov_w).reshape((config.n_fov_h, config.n_fov_w), order='F')

    # Arrangement of the FOVs - default is ul
    if config.start_corner == "ur":
        order = np.flip(order, 1)
    elif config.start_corner == "bl":
        order = np.flip(order, 0)
    elif config.start_corner == "br":
        order = np.flip(order, (0,1))
    
    print("FOV ordering")
    print(order)

    stitched = np.zeros((fov_h*config.n_fov_h, fov_w*config.n_fov_w), dtype=fov_dtype)

    for i_fov in range(n_fov):
        coord = np.where(order==i_fov)
        h_idx = coord[0][0]
        w_idx = coord[1][0]
        h_start = h_idx*fov_h
        w_start = w_idx*fov_w
        h_end = h_start + fov_h
        w_end = w_start + fov_w

        fov_num = i_fov + config.min_fov
        
        # All files for FOV 
        pattern_fov = get_string_with_pattern(fov_num, config.pattern_f)
        print(pattern_fov)
        found_fov = [check_pattern(s, pattern_fov) for s in fp_dapi_list]
        fp_stack_fov = [fp_dapi_list[i] for i,x in enumerate(found_fov) if x]
       
        # Take MIP - or z level 
        if config.mip:
            dapi_stack = np.zeros((len(fp_stack_fov), fov_h, fov_w), dtype=fov_dtype)
            for i, fp in enumerate(fp_stack_fov):
                dapi_stack[i,:,:] = read_dapi(fp, 
                                              config.channel_first, 
                                              config.channel_dapi)

            fov_img = np.max(dapi_stack, axis=0)

        else:
            # Find z level slice for FOV  
            pattern_slice = get_string_with_pattern(config.z_level, config.pattern_z)
            print(pattern_slice)
            found_slice = [check_pattern(s, pattern_slice) for s in fp_stack_fov]
            found_slice_idx = [i for i, x in enumerate(found_slice) if x]
            if len(found_slice_idx) > 1:
                sys.exit(f"Found {len(found_slice_idx)} files with {pattern_slice} for FOV {fov_num}")
            
            print(fp_stack_fov[found_slice_idx[0]])
            fov_img = read_dapi(fp_stack_fov[found_slice_idx[0]],
                                config.channel_first,
                                config.channel_dapi)
            
        # Flip
        if config.flip_ud:
            fov_img = np.flip(fov_img, 0)
                
        # Place into appropriate location in stitched image 
        stitched[h_start:h_end, w_start:w_end] = fov_img.copy()
        
    # Save 
    fp_output = dir_dataset+'/'+config.fp_dapi_stitched
    tifffile.imwrite(fp_output, stitched, photometric='minisblack')
    print(f"Saved {fp_output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../../data/', type=str)
    parser.add_argument('--dataset', default='dataset_cosmx_nsclc', type=str)
    parser.add_argument('--dir_dapi', default='Lung5_Rep1-RawMorphologyImages', type=str)
    parser.add_argument('--ext_dapi', default='tif', type=str)

    parser.add_argument('--pattern_z', default='Z###', type=str, help="String pattern to find in the file names for the Z number, or None for no Z component")
    parser.add_argument('--pattern_f', default='F###', type=str, help="String pattern to find in file names for the FOV number")

    parser.add_argument('--channel_first', action='store_true', help="channel axis first or last in image volumes")
    parser.set_defaults(channel_first=False)
    parser.add_argument('--channel_dapi', default='-1', type=int, help="channel index DAPI image is in")

    parser.add_argument('--fp_dapi_stitched', default='dapi_preprocessed.tif', type=str)
    
    parser.add_argument('--n_fov', default=30, type=int, help="total number of FOVs")        
    parser.add_argument('--min_fov', default=1, type=int, help="smallest FOV number - usually 0 or 1")        
    parser.add_argument('--n_fov_h', default=6, type=int, help="number of FOVs tiled along vertical axis")        
    parser.add_argument('--n_fov_w', default=5, type=int, help="number of FOVs tiled along horizontal axis")        
        
    parser.add_argument('--start_corner', default='ul', type=str, help="location of first FOV - choose from ul, ur, bl, br")

    parser.add_argument('--row_major', action='store_true', help="row major ordering of FOVs")
    parser.set_defaults(row_major=False)

    parser.add_argument('--z_level', default=1, type=int, help="which z slice to use, or --mip to use MIP")        

    parser.add_argument('--mip', action='store_true', help="take the maximum intensity projection across all Z")
    parser.set_defaults(mip=False)
    
    parser.add_argument('--flip_ud', action='store_true', help="flip images up/down before stitching")
    parser.set_defaults(flip_ud=False)
    
    config = parser.parse_args()
    main(config)