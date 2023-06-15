# BIDCell: Biologically-informed self-supervised learning for segmentation of subcellular spatial transcriptomics data

For more details, please refer to our paper: https://doi.org/10.1101/2023.06.13.544733

Recent advances in subcellular imaging transcriptomics platforms have enabled spatial mapping of the expression of hundreds of genes at subcellular resolution and provide topographic context to the data. This has created a new data analytics challenge to correctly identify cells and accurately assign transcripts, ensuring that all available data can be utilised. To this end, we introduce BIDCell, a self-supervised deep learning-based framework that incorporates cell type and morphology information via novel biologically-informed loss functions. We also introduce CellSPA, a comprehensive evaluation framework consisting of metrics in five complementary categories for cell segmentation performance. We demonstrate that BIDCell outperforms other state-of-the-art methods according to many CellSPA metrics across a variety of tissue types of technology platforms, including 10x Genomics Xenium. Taken together, we find that BIDCell can facilitate single-cell spatial expression analyses, including cell-cell interactions, enabling great potential in biological discovery.

![alt text](Figure1.png)

## Installation

> **Note**: A GPU with 12GB VRAM is strongly recommended for the deep learning component, and 32GB RAM for data processing.
We ran BIDCell on a Linux system with a 12GB NVIDIA GTX Titan V GPU, Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz with 16 threads, and 64GB RAM.

1. Clone repository:
    
        git clone https://github.com/SydneyBioX/BIDCell.git

2. Create virtual environment:
    
        conda create --name BIDCell python=3.7
    
3. Activate virtual environment:
    
        conda activate BIDCell

4. Install dependencies:
    
        pip install -r requirements.txt

        pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html


## Datasets and Preprocessing

Currently, our repository provides the processed single cell reference for breast cancer, with the positive and negative markers. We also provide the nuclei segmentation and nuclei cell-type classifications for a public dataset. We will be including instructions for performing these preprocessing tasks for other datasets shortly.

Unzip the provided nuclei segmentation ``data/nuclei.zip`` and place the image as such: ``data/nuclei.tif``

To train and run BIDCell, download the dataset (Xenium Output Bundle In Situ Replicate 1) from https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast 


### Process the transcript data

1. Put ``transcripts.csv.gz`` from the Xenium Output Bundle into the ``/preprocess`` folder, or note its path.

2. Convert detected transcripts to image maps of gene expressions:

		cd preprocess
	
	then, 
	
        python generate_expr_maps.py

    or,

        python generate_expr_maps.py --fp_transcripts /PATH/TO/transcripts.csv.gz --n_processes NUM_CPUS

    If you receive the error: ``pickle.UnpicklingError: pickle data was truncated``, try reducing NUM_CPUS

    By default, the maps will be stored in ``/data/expr_maps``

3. Split expression maps to patches for the deep learning model:

        python split_expr_maps_to_patches.py
    
    By default, ``--patch_size`` is set to 48.


## Running BIDCell:

Make sure the provided nuclei segmentation ``data/nuclei.zip`` has been extracted and you have ``data/nuclei.tif`` 

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
	
If you receive the error: ``pickle.UnpicklingError: pickle data was truncated``, try reducing NUM_CPUS


## Citation

If BIDCell has assisted you with your work, please kindly cite our paper:

Fu, X., Lin, Y., Lin, D., Mechtersheimer, D., Wang, C., Ameen, F., Ghazanfar, S., Patrick, E., Kim, J., & Yang, J. Y. H. (2023). Biologically-informed self-supervised learning for segmentation of subcellular spatial transcriptomics data. bioRxiv, 2023.2006.2013.544733. https://doi.org/10.1101/2023.06.13.544733