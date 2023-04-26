# BIDCell: Biologically-informed deep learning for cell segmentation of subcelluar spatial transcriptomics data 

TODO: add abstract

For more details, please refer to our paper (TODO: link to paper).

TODO: Add fig 1a


## Installation

1. Clone repository:
    
        git clone TODO: add to github 

2. Create virtual environment:
    
        conda create --name BIDCell python=3.7
    
3. Activate virtual environment:
    
        conda activate BIDCell

4. Install dependencies:
    
        pip install -r requirements.txt

        pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html


## Datasets and Preprocessing

Currently, our repository provides the processed single cell reference for breast cancer, with the positive and negative markers. We also provide the nuclei segmentation and nuclei cell-type classifications for a public dataset. We will be including instructions for performing these preprocessing tasks for other datasets shortly.

To train and run BIDCell, first download the dataset (Xenium Output Bundle In Situ Replicate 1) from https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast 

Unzip ``data/nuclei.zip`` and place the image as such: ``data/nuclei.tif``


### Process the transcript data

1. Put ``transcripts.csv.gz`` into the ``/preprocess`` folder, or note its path.

2. Convert detected transcripts to image maps of gene expressions:

		cd preprocess
	
	then, 
	
        python generate_expr_maps.py

    or,

        python generate_expr_maps.py --fp_transcripts /PATH/TO/transcripts.csv.gz --n_processes NUM_CPUS

    By default, the maps will be stored in ``/data/expr_maps``

3. Split expression maps to patches for the deep learning model:

        python split_expr_maps_to_patches.py
    
    By default, ``--patch_size`` is set to 48.


## Running BIDCell:

    cd BIDCell_model


### Training the model

    python train.py

Hyperparameters are defined in ``/configs/config.json``, such as the weight of each type of loss function. 

To specify the config file:

    python train.py --config_file configs/config.json


### Predicting from the trained model

Specify ``--test_epoch`` and ``--test_step`` of the saved model to generate predictions. 

    python predict.py --config_file configs/config.json --test_epoch 1 --test_step 4000


### Postprocessing segmentation predictions

    python postprocess_predictions.py --dir_id last --epoch 1 --step 4000 --nucleus_fp ../data/nuclei.tif

    or, specify name of directory under /BIDCell/experiments/, e.g.: 

    python postprocess_predictions.py --dir_id 2023_April_18_19_31_46 --epoch 1 --step 4000 --nucleus_fp ../data/nuclei.tif


### Extracting cell expressions

To extract the gene expressions of segmented cells: 

    cd analysis
	
    python extract_cell_expressions.py --fp_seg /PATH/TO/SEGMENTATION.tif --fp_transcripts /PATH/TO/transcripts.csv.gz --output_dir /DIR_NAME --n_processes NUM_CPUS

For example,
	
    python extract_cell_expressions.py --fp_seg ../BIDCell_model/experiments/2023_April_18_19_31_46/test_output/epoch_1_step_4000_connected.tif --fp_transcripts ../preprocess/transcripts.csv.gz --output_dir cell_gene_matrices/2023_April_18_19_31_46
	

## Citation

If BIDCell has assisted you with your work, please kindly cite our paper:

- TODO: add citation