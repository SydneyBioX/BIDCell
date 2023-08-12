# BIDCell: Biologically-informed self-supervised learning for segmentation of subcellular spatial transcriptomics data

> **Note**: This version of the code was tested across 4 major platforms, including Xenium, CosMx, MERSCOPE, and Stereo-seq. Please take care with the appropriate arguments that need to be provided. Also, we are in the the process of making a Python package for BIDCell.

For more details of our method, please refer to: https://doi.org/10.1101/2023.06.13.544733

Recent advances in subcellular imaging transcriptomics platforms have enabled spatial mapping of the expression of hundreds of genes at subcellular resolution and provide topographic context to the data. This has created a new data analytics challenge to correctly identify cells and accurately assign transcripts, ensuring that all available data can be utilised. To this end, we introduce BIDCell, a self-supervised deep learning-based framework that incorporates cell type and morphology information via novel biologically-informed loss functions. We also introduce CellSPA, a comprehensive evaluation framework consisting of metrics in five complementary categories for cell segmentation performance. We demonstrate that BIDCell outperforms other state-of-the-art methods according to many CellSPA metrics across a variety of tissue types of technology platforms, including 10x Genomics Xenium. Taken together, we find that BIDCell can facilitate single-cell spatial expression analyses, including cell-cell interactions, enabling great potential in biological discovery.

![alt text](Figure1.png)

## Installation

> **Note**: A GPU with at least 12GB VRAM is strongly recommended for the deep learning component, and 32GB RAM for data processing.
We ran BIDCell on a Linux system with a 12GB NVIDIA GTX Titan V GPU, Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz with 16 threads, and 64GB RAM.

1. Clone repository:
    
        git clone https://github.com/SydneyBioX/BIDCell.git

2. Create virtual environment:
    
        conda create --name BIDCell python=3.9
    
3. Activate virtual environment:
    
        conda activate BIDCell

4. Install dependencies:
    
        pip install -r requirements.txt

    Installation of dependencies typically requires a few minutes. 


## Datasets

Under `data` create a folder for each dataset. In this folder, place the transcripts file and the DAPI-stained images. The preprocessed data and the extracted cell-gene matrices will be stored in this folder.

Example folder structure and files for different datasets/platforms:

```
├── BIDCell
│   ├── bidcell
│   │   ├── ...
│   ├── data
│   │   ├── dataset_xenium_breast1
|   │   │   ├── morphology_mip.ome.tif
|   │   │   ├── transcripts.csv.gz
│   │   ├── dataset_cosmx_nsclc
|   │   │   ├── Lung5_Rep1-RawMorphologyImages
|   |   │   │   ├── *.TIF
|   │   │   ├── Lung5_Rep1_tx_file.csv
│   │   ├── dataset_merscope_melanoma2
|   │   │   ├── HumanMelanomaPatient2_images_mosaic_DAPI_z0.tif
|   │   │   ├── HumanMelanomaPatient2_detected_transcripts.csv
│   │   ├── dataset_stereoseq_mousebrain
|   │   │   ├── Mouse_brain_Adult.tif
|   │   │   ├── Mouse_brain_Adult_GEM_bin1.tsv.gz
|   │   │   ├── selected_genes.txt
│   │   ├── sc_references
|   │   │   ├── sc_breast.csv
|   │   │   ├── sc_breast_markers_pos.csv
|   │   │   ├── sc_breast_markers_neg.csv
|   │   │   ├── ...
│   │   ├── transcripts_to_filter.txt
│   ├── ...
```

For example, the Xenium breast cancer dataset (Xenium Output Bundle In Situ Replicate 1) may be downloaded from https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast. Place `morphology_mip.ome.tif` and `transcripts.csv.gz` under `data/dataset_xenium_breast1`


## Single-cell reference and markers

BIDCell uses single-cell reference data for improved results. These can be downloaded from public repositories such as TISCH2, Allen Brain Map, and the Human Cell Atlas. 

Please see the provided breast cancer single-cell reference and positive/negative marker files (`sc_breast.csv`, `sc_breast_markers_pos.csv`, and `sc_breast_markers_neg.csv`) as a template. Ensure that the column names (genes) are naturally sorted (e.g., with `natsort()`).

The reference csv file contains average expressions for all of the genes in the spatial transcriptomic dataset of different cell types. You may choose an appropriate list of cell types to include for your data.

The positive and negative markers files contain the respective marker genes for each cell type. The positive and negative markers were those with expressions in the highest and lowest 10 percentile for each cell type of a tissue sample. We found that removing positive markers that were common to at least a third of cell types in each dataset was appropriate across various datasets. Using a larger number of positive markers tends to increase the size of predicted cells. Manual curation and alternative approaches to determine the marker genes can also be used.

Only <1,000 genes are needed to perform segmentation. Specify a selection of genes in a file `selected_genes.txt` and place in the dataset folder (see Stero-seq example).


## Running example dataset

After installation, downloading the Xenium breast cancer dataset and placing the transcripts and DAPI files in appropriate locations (see [Datasets](#datasets)), run the example dataset:

        python example_xenium.py


## Running BIDCell for your dataset

Several steps are involved in the entire pipeline to generate segmentations and cell-gene matrices from DAPI images and detected transcripts. These steps can be grouped into:

1. Data preprocessing
    - Nuclei segmentation
    - Preparing transcriptomic images and patches
    - Preannotation of nuclei
2. Model training
3. Prediction and extracting cell expressions

### Config files

In `bidcell/model/configs` create a config file, e.g., `config_cosmx_nsclc.json`
- Update `data_sources` for current dataset
- Specify `cell_types`
- Specify `elongated` cell types
- The weights of losses typically can all be left at 1.0, but if cells are too small or big, try adjusting `cc_weight` to 0.5 (to reduce size) or 2.0 (to increase size)

### Setting up the script to run all steps

Example scripts are provided for different datasets/platforms to run all steps. BIDCell may also be applied to data from other technologies such as MERFISH. The format of different datasets varies. Please take note of the arguments:

- `dataset`: name of dataset
- `fp_dapi`: name of DAPI image
- `fp_transcripts`: name of transcripts file
- `x_col`: name of x location column in transcripts file
- `y_col`: name of y location column in transcripts file
- `gene_col`: name of genes column in transcripts file
- `fp_config`: name of config file under `bidcell/model/configs`
- `n_processes`: number of CPUs for multiprocessing

If you receive the error: ``pickle.UnpicklingError: pickle data was truncated``, try reducing `--n_processes`

Scaling and alignment arguments:
- `target_pix_um`: microns per pixel to perform segmentation 
- `base_pix_x`: convert to microns along width by multiplying the original pixels by base_pix_x microns per pixel
- `base_pix_y`: convert to microns along height by multiplying the original pixels by base_pix_y microns per pixel
- `base_ts_x`: convert between transcript locations and target pixels along width
- `base_ts_y`: convert between transcript locations and target pixels along height
- `global_shift_x`: additional adjustment to align transcripts to DAPI in target pixels along image width
- `global_shift_y`: additional adjustment to align transcripts to DAPI in target pixels along image height
- `patch_size`: size of input patches to segmentation model

Performing segmentation at a higher resolution requires a larger patch size, thus more GPU memory. 


### Additional information

Expected outputs:
- .tif file of segmented cells, where the value corresponds to cell IDs
- .csv file of gene expressions of segmented cells

Expected runtime (based on our system for the Xenium breast cancer dataset):
- Training: ~10 mins for 4,000 steps
- Inference: ~ 50 mins
- Postprocessing: ~ 30 mins


## Contact us

If you have any enquiries, especially about using BIDCell to segment cells in your data, please contact xiaohang.fu@sydney.edu.au. We are also happy to receive any suggestions and comments.


## Citation

If BIDCell has assisted you with your work, please kindly cite our paper:

Fu, X., Lin, Y., Lin, D., Mechtersheimer, D., Wang, C., Ameen, F., Ghazanfar, S., Patrick, E., Kim, J., & Yang, J. Y. H. (2023). Biologically-informed self-supervised learning for segmentation of subcellular spatial transcriptomics data. bioRxiv, 2023.2006.2013.544733. https://doi.org/10.1101/2023.06.13.544733