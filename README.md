# Sequence models to predict enhancers in different tissues of the Drosophila embryo using transfer learning

*<ins>Targeted design of synthetic enhancers for selected tissues in the Drosophila embryo</ins>*  
Bernardo P. de Almeida, Christoph Schaub, Michaela Pagani, Stefano Secchia, Eileen E. M. Furlong, Alexander Stark. 2023

This repository contains the code used to to train the models as well as to make predictions on new sequences.

## Training of DNA accessibility models

Data used to train and evaluate the accessibility models as well as the final trained models are available on zenodo at https://doi.org/10.5281/zenodo.8011696.

To train models across 10 genomic folds and evaluate them, download the training data (Accessibility_models_training_data) to Accessibility_models/Data folder and run:
```
Accessibility_models/Run_models.sh
```
This script will train 3 replicate models per genomic fold and evaluate it in the test set. It will also compute the contribution scores over the peaks of the respective tissue and run TF-Modisco on those scores to discover predictive motifs.

Trained models and evaluation results can be found in Accessibility_model_files.tar.gz and Accessibility_models_test_set_predictions.rds.

## Training of enhancer activity transfer-learning models

Data used to train and evaluate the enhancer activity models as well as the final trained models are available on zenodo at https://doi.org/10.5281/zenodo.8011696.

To train models across 10 genomic folds and evaluate them, download the training data (EnhancerActivity_model_files) to Enhancer_activity_models/Data folder and run:
```
Enhancer_activity_models/Run_models.sh
```
This script will train 3 replicate models per genomic fold and evaluate it in the test set. It will also compute the contribution scores over the peaks of the respective tissue.

Trained models and evaluation results can be found in EnhancerActivity_model_files.tar.gz and EnhancerActivity_models_results_per_tissue_test_set.rds.

## Prediction for new DNA sequences
To predict the accessibility levels or enhancer activity score in a given tissue of the Drosophila embryo for new DNA sequences, please run:
```
# Clone this repository
git clone https://github.com/bernardo-de-almeida/DeepSTARR.git
cd DeepSTARR/DeepSTARR

# download a DNA-accessibility or enhancer-activity model from zenodo (https://doi.org/10.5281/zenodo.8011696)
# example with DNA-accessibility model for CNS fold01 replicate1 in Accessibility_model_files --> Results_fold01_CNS_DeepSTARR2_rep1/Model*

# create 'DeepSTARR' conda environment by running the following:
conda create --name DeepSTARR python=3.7 tensorflow=1.14.0 keras=2.2.4 # or tensorflow-gpu/keras-gpu if you are using a GPU
source activate DeepSTARR
pip install git+https://github.com/AvantiShri/shap.git@master
pip install 'h5py<3.0.0'
pip install deeplift==0.6.13.0

# Run prediction script on fasta files with 1,001 bp sequences
python Accessibility_models/Predict_CNN_model_from_fasta.py \
  -s Sequences_example.fa \
  -m Results_fold01_CNS_DeepSTARR2_rep1/Model \
  -o Sequences_example

```
Where:
* -s FASTA file with input 1,001 bp DNA sequences
* -m model file (from accessibility or enhancer activity model)
* -o output directory

We recommend using the models from the different folds and average the prediction scores for a more robust prediction.

## Questions
If you have any questions/requests/comments please contact me at [bernardo.almeida94@gmail.com](mailto:bernardo.almeida94@gmail.com).
