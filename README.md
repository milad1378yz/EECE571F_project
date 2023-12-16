# EECE571F_project

## Overview
This repository hosts the code developed for the EECE-571F course at the University of British Columbia. The project focuses on deep learning applications in medical imaging, specifically in fetal ultrasound image analysis.

## Authors
- Pooria Ashrafian
- Milad Yazdani

## Dataset Preparation
To use the dataset for this project, follow these steps:
1. Download the dataset from [Zenodo](https://zenodo.org/records/3904280).
2. Extract and place the `FETAL_PLANES_ZENODO` folder in the `data` directory of this repository.

## Installation
Install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```

# Unconditional Model Training
To train the unconditional model, navigate to the unconditional directory and execute the following command:

```bash
cd unconditional
python train_unconditional.py --class_ultra <class_unltrasound> --num_epochs 400
```
Replace <class_unltrasound> with one of the following classes:

* Fetal brain_Trans-cerebellum
* Fetal brain_Trans-thalamic
* Fetal brain_Trans-ventricular
* Fetal abdomen
* Fetal femur
* Fetal thorax
Also, if you are in Sockeye server you can use the 6 classes of .sh file in this folder.

The trained model will be saved in the unconditional/diffusion_models directory. For comprehensive evaluation, ensure all six classes are trained.

## Data Generation Unconcditional
To generate data using the trained models, run:

```bash
python data_generation.py --number_of_images 200
```
Also, you can use data_generation.sh in this directory. if you are working with Sockeye.
Generated images will be saved in the generated folder.


# Conditional Model Training - Textual

## Overview
This section details the fine-tuning of the model using textual prompts. We utilize the pre-trained [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) model for this purpose.

## Dataset Preparation -Texual
First, prepare a dataset suitable for this project by creating a `metadata.csv` file. To do this, follow these steps:
```bash
cd data
python dataset_builder.py
```
* Note: Do not use the --do_random_string flag in this part.

## Model Training
To train the Stable Diffusion model capable of generating images for all classes, use the script provided below:

```bash
cd text2image
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="data/FETAL_PLANES_ZENODO"
export OUTPUT_DIR="results/"

accelerate launch  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=30000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit=3 \
  --resume_from_checkpoint="latest" \
```

Additionally, you can use `trainer.sh` if you are working in the Sockeye environment.

## Data Generation - Texual
To generate synthetic dataset you can use the following script:
```bash
python data_generation.py --number_of_images 200
```
* Note: Do not use the --do_random_string flag in this part.
Also if you are working in the Sockeye you can use data_generation script.
# Conditional Model Training - Abstract
In this part we fine-tune the model using abtract prompts.
The only differences are in Dataset Preparation and Data Generation parts.

## Dataset Preparation - Abtract
```bash
cd data
python dataset_builder.py --do_random_string
```

## Data Generation - Abstract
```bash
python data_generation.py --number_of_images 200 --do_random_string
```

# Evaluation

## Metric Evaluations
### Overview
We employ two metrics, FID (Fréchet Inception Distance) and KID (Kernel Inception Distance), to evaluate the synthetic images.

### Data Separation
First, separate the real data using the following commands:
```bash
cd data
python separator.py
```
This script will create a `separated_data` folder within the data directory.

### Metric Calculation
To calculate FID and KID scores, execute the following:
```bash
cd ../evaluate
python fid_kid.py --dir_generator <path to the generated data directory> --dir_separated_data "../data/separated_data"
```
Replace `<path to the generated data directory>` with the path where the generated data is stored.

## Classification Evaluation
### Overview
We utilize a pre-trained ResNet50 model, modifying its last layer for fine-tuning and classification on our dataset. This process evaluates the effectiveness of data augmentation using diffusion models.

### Running the Classifier
To assess the classifier's performance, use the script below:
```bash
cd classifier
python train_classifier.py --data_dir "../data/separated_data" --new_data_dir <path to generated data directory>
```
If `<path to generated data directory>` is not provided, the classifier will train only on real data.


