import collections
import datetime as dt
import json
import os
import random
import re
import sys

import matplotlib.pyplot as plt
import natsort
import numpy as np
from scipy.special import softmax


def sorted_alphanumeric(data):
    """
    Alphanumerically sort a list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def make_dir(dir_path):
    """
    Make directory if doesn't exist
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def delete_file(path):
    """
    Delete file if exists
    """
    if os.path.exists(path):
        os.remove(path)


def get_files_list(path, ext_array=[".tif"]):
    """
    Get all files in a directory with a specific extension
    """
    files_list = list()
    dirs_list = list()

    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if any(x in file for x in ext_array):
                files_list.append(os.path.join(root, file))
                folder = os.path.dirname(os.path.join(root, file))
                if folder not in dirs_list:
                    dirs_list.append(folder)

    return files_list, dirs_list


def json_file_to_pyobj(filename):
    """
    Read json config file
    """

    def _json_object_hook(d):
        return collections.namedtuple("X", d.keys())(*d.values())

    def json2obj(data):
        return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def get_newest_id(exp_dir="model_outputs"):
    """Get the latest experiment ID based on its timestamp

    Parameters
    ----------
    exp_dir : str, optional
        Name of the directory that contains all the experiment directories

    Returns
    -------
    exp_id : str
        Name of the latest experiment directory
    """
    folders = next(os.walk(exp_dir))[1]
    if len(folders) == 0:
        sys.exit(f"No model output folders found in {exp_dir}")
    folders = natsort.natsorted(folders)
    folder_last = folders[-1]
    exp_id = folder_last.replace("\\", "/")
    return exp_id


def get_experiment_id(make_new, dir_id, data_dir):
    """
    Get timestamp ID of current experiment
    """
    if make_new is False:
        if dir_id == "last":
            timestamp = get_newest_id(os.path.join(data_dir, "model_outputs"))
        else:
            timestamp = dir_id
    else:
        timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    return timestamp


def get_seg_mask(sample_seg, sample_n):
    """
    Generate the segmentation mask with unique cell IDs
    """
    sample_n = np.squeeze(sample_n)

    # Background prob is average probability of all cells EXCEPT FOR NUCLEI
    sample_probs = softmax(sample_seg, axis=1)
    bgd_probs = np.expand_dims(np.mean(sample_probs[:, 0, :, :], axis=0), 0)
    fgd_probs = sample_probs[:, 1, :, :]
    probs = np.concatenate((bgd_probs, fgd_probs), axis=0)
    final_seg = np.argmax(probs, axis=0)

    # Map predictions to original cell IDs
    ids_orig = np.unique(sample_n)
    if ids_orig[0] != 0:
        ids_orig = np.insert(ids_orig, 0, 0)
    ids_pred = np.unique(final_seg)
    if ids_pred[0] != 0:
        ids_pred = np.insert(ids_pred, 0, 0)
    ids_orig = ids_orig[ids_pred]

    dictionary = dict(zip(ids_pred, ids_orig))
    dictionary[0] = 0
    final_seg_orig = np.copy(final_seg)
    final_seg_orig = np.vectorize(dictionary.get)(final_seg)

    # Add nuclei back in
    final_seg_orig = np.where(sample_n > 0, sample_n, final_seg_orig)

    return final_seg_orig


def save_fig_outputs(sample_seg, sample_n, sample_sa, sample_expr, patch_fp):
    """
    Generate figure of inputs and outputs
    """
    sample_n = np.squeeze(sample_n)

    sample_expr = np.squeeze(sample_expr)
    sample_expr[sample_expr > 0] = 1

    sample_sa = np.squeeze(np.sum(sample_sa, 0))

    final_seg_orig = get_seg_mask(sample_seg, sample_n)

    # Randomise colours for plot
    cells_ids_orig = np.unique(final_seg_orig)
    n_cells_ids = len(cells_ids_orig)
    cell_ids_rand = np.arange(1, n_cells_ids + 1)
    random.shuffle(cell_ids_rand)
    dictionary = dict(zip(cells_ids_orig, cell_ids_rand))
    dictionary[0] = 0
    final_seg_mapped = np.copy(final_seg_orig)
    final_seg_mapped = np.vectorize(dictionary.get)(final_seg_orig)
    nuclei_mapped = np.copy(sample_n)
    nuclei_mapped = np.vectorize(dictionary.get)(sample_n)

    # Plot
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(nuclei_mapped, cmap=plt.cm.nipy_spectral)
    ax[0].set_title("Nuclei")
    ax[1].imshow(final_seg_mapped, cmap=plt.cm.nipy_spectral)
    ax[1].set_title("Cells")
    ax[2].imshow(sample_expr, cmap=plt.cm.gray)
    ax[2].set_title("Expressions")
    # ax[3].imshow(sample_sa, cmap=plt.cm.gray)
    # ax[3].set_title("Eligible")

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    # plt.show()

    fig.savefig(patch_fp)
    plt.close(fig)
