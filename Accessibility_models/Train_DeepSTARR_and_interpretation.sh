#!/bin/bash
set -o errexit
set -o pipefail

######
# Bernardo Almeida (2022)
######


################################################################################
# Set default values
################################################################################

ContrScores=0
ContrScores_peaks="Data/Sequences_all_peaks_1kb.fa"
Modisco=0

Script_dir=Accessibility_models/

################################################################################
# Help
################################################################################

if [ $# -eq 0 ]; then
  echo >&2 "
$(basename $0) - Train DeepSTARR single task model + contribution scores + TF-Modisco

USAGE: $(basename $0) -d <fold file name> -f <Sequences fasta> -a <architecture> -v <cell type output> -o <results output path> [OPTIONS]

 -d     Input ID to get fasta file and sequences txt file         [ required ]
 -f     All sequences fasta file to evaluate predictions          [ required ]
 -a     Model architecture                                        [ required ]
 -v     Variable to predict                                       [ required ]
 -o     Output directory name                                     [ required ]
 -c     Run nucl contr scores (0/1)                               [default: $ContrScores]
 -p     Peaks to generate nucl contr scores                       []
 -m     Run TF-Modisco (0/1)                                      [default: $Modisco]
 
"
  exit 1
fi

################################################################################
# Parse input and check for errors
################################################################################

while getopts "d:f:v:a:o:c:p:m:" o
do
    case "$o" in
        d) Input_ID="$OPTARG";;
        f) all_seq_fasta="$OPTARG";;
        v) variable_output="$OPTARG";;
        a) arch="$OPTARG";;
        o) OUTDIR="$OPTARG";;
        c) ContrScores="$OPTARG";;
        p) ContrScores_peaks="$OPTARG";;
        m) Modisco="$OPTARG";;
        \?) exit 1;;
  esac
done


echo
echo Input_fasta: $Input_ID
echo all_seq_fasta: $all_seq_fasta
echo variable_output: $variable_output
echo Architecture: $arch
echo Output director: $OUTDIR
echo ContrScores: $ContrScores
echo Peaks for contribution scores: $ContrScores_peaks
echo TF-Modisco: $Modisco
echo

### create output directory
mkdir -p $OUTDIR

################################################################################
# Train model
################################################################################

echo
echo "Training model ..."
echo

mkdir -p $OUTDIR/log_training
bin/my_bsub_gridengine -P g -G "gpu:1" -m 120 -T '03:00:00' -o $OUTDIR/log_training -n Training "$Script_dir/Train_model.py -i $Input_ID -v $variable_output -a $arch -o $OUTDIR/Model" > $OUTDIR/log_training/msg.model_training.tmp

# get job IDs to wait for mapping to finish
ID_main=$(paste $OUTDIR/log_training/msg.model_training.tmp | grep Submitted | awk '{print $4}')

################################################################################
# Predictions for evaluation
################################################################################

echo
echo "Predictions of sets for evaluation ..."
echo

# predict + evaluation scatter plots

pred_script=$Script_dir/Predict_CNN_model_from_fasta.py
model=$OUTDIR/Model
all_seq_fasta_name=$(basename $all_seq_fasta)
Input_ID_name=$(basename $Input_ID)
bin/my_bsub_gridengine -d "$ID_main" -n predict -o $OUTDIR/log_predictions -m 60 -T '02:00:00' "$pred_script -s $all_seq_fasta -m $model -o $OUTDIR; \
  $Script_dir/Model_evaluation_all_regions.R Data/All_bins_information.rds $Input_ID_name $variable_output $OUTDIR/${all_seq_fasta_name}_predictions_Model.txt"

# predictions for all peaks - needed below for contr scores
for seq in ${ContrScores_peaks//,/ }; do
  bin/my_bsub_gridengine -d "$ID_main" -n predict -o $OUTDIR/log_predictions -T 40:00 "$pred_script -s $seq -m $model -o $OUTDIR"
done


################################################################################
# Nucleotide contribution scores
################################################################################

if [ "$ContrScores" == "1" ]; then

  echo
  echo "Nucleotide contribution scores ..."
  echo

  mkdir -p $OUTDIR/log_DeepExplainer
  for seq in ${ContrScores_peaks//,/ }; do
    seq_name=$(basename $seq)
    bin/my_bsub_gridengine -d "$ID_main" -m 40 -T '04:00:00' -Q medium -P g -G "gpu:1" -o $OUTDIR/log_DeepExplainer -n DeepExplainer "$Script_dir/run_DeepSHAP_DeepExplainer.py -m $OUTDIR/Model -s $seq -b dinuc_shuffle" > $OUTDIR/log_DeepExplainer/msg.contr_scores_${seq_name}.tmp

    # Convert contr scores to R object
    ID_contrscores=$(paste $OUTDIR/log_DeepExplainer/msg.contr_scores_${seq_name}.tmp | grep Submitted | awk '{print $4}') # contrscores usually take longer than predictions above, that are also needed
    bin/my_bsub_gridengine -d "$ID_contrscores" -m 20 -T '02:00:00' -o $OUTDIR/log_DeepExplainer -n DeepExplainer_convert "$Script_dir/Load_and_average_contr_scores.R $variable_output $OUTDIR/Model $seq $OUTDIR/Sequences_all_peaks_1kb.fa_predictions_Model.txt"
  done

fi



################################################################################
# TF-Modisco
################################################################################

if [ "$Modisco" == "1" ]; then

  echo
  echo "TF-Modisco ..."
  echo

  mkdir -p $OUTDIR/log_modisco
  for seq in ${ContrScores_peaks//,/ }; do
    seq_name=$(basename $seq)
    ID_contrscores=$(paste $OUTDIR/log_DeepExplainer/msg.contr_scores_${seq_name}.tmp | grep Submitted | awk '{print $4}')
    imp_score=$OUTDIR/Model_${seq_name}_dinuc_shuffle_deepSHAP_DeepExplainer_importance_scores.h5
    bin/my_bsub_gridengine -d "$ID_contrscores" -m 40 -T '24:00:00' -Q medium -P g -G "gpu:1" -o $OUTDIR/log_modisco -n modisco "$Script_dir/TFmodisco.py $seq $imp_score $OUTDIR/${seq_name}" > $OUTDIR/log_modisco/msg.modisco_${seq_name}.tmp

    
  done



fi






