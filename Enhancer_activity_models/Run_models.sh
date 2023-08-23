
### Run main model across folds and output variables

mkdir Enhancer_activity_models/Models/

Script_dir=Enhancer_activity_models/

fold_list="fold01,fold02,fold03,fold04,fold05,fold06,fold07,fold08,fold09,fold10"
celltype_IDs="brain_VNC,muscle,epidermis,Gut,brain_spec"

for fold in ${fold_list//,/ }; do
	for var in ${celltype_IDs//,/ }; do
  	  	for rep in 1 2 3; do
  	  	
  	  	  # DNA accessibility models to initialize
  	  	  if [ "$var" == "brain_VNC" ]; then arch=Accessibility_models/Models/Results_${fold}_CNS_DeepSTARR2_rep${rep}/Model; fi
          if [ "$var" == "brain_spec" ]; then arch=Accessibility_models/Models/Results_${fold}_brain_DeepSTARR2_rep${rep}/Model; fi
          if [ "$var" == "epidermis" ]; then arch=Accessibility_models/Models/Results_${fold}_epidermis_DeepSTARR2_rep${rep}/Model; fi
          if [ "$var" == "muscle" ]; then arch=Accessibility_models/Models/Results_${fold}_visceralmuscle_DeepSTARR2_rep${rep}/Model; fi # visceral accessibility seem the most important
          if [ "$var" == "Gut" ]; then arch=Accessibility_models/Models/Results_${fold}_midgut_DeepSTARR2_rep${rep}/Model; fi
          
          OUTDIR=Enhancer_activity_models/Models/Results_${fold}_${var}_rep${rep}
        	mkdir -p $OUTDIR/log_training
      		
      		JOB_ID=${fold}_${var}_rep${rep}
      
      		bin/my_bsub_gridengine -P g -G "gpu:1" -m 50 -T '01:00:00' -o $OUTDIR/log_training -n Training_${JOB_ID} "Enhancer_activity_models/Train_transfer_learning_model.py -i Data/$fold -v $var -a $arch -o $OUTDIR/Model" > $OUTDIR/log_training/msg.model_training.tmp
      
      		# get job IDs to wait for mapping to finish
      		ID_main=$(paste $OUTDIR/log_training/msg.model_training.tmp | grep Submitted | awk '{print $4}')
      
      		# Predict for merged peaks
      		seq=Accessibility_models/Data/Sequences_merged_peaks_1kb.fa
      		seq_name=$(basename $seq)
          bin/my_bsub_gridengine -d "$ID_main" -n predict_peaks_${JOB_ID} -o $OUTDIR/log_predictions -m 10 -T '1:00:00' "$Script_dir/Predict_CNN_model_from_fasta.py -s $seq -m $OUTDIR/Model -o $OUTDIR"
          
          # Predict for final test set
      		seq=Enhancer_activity_models/Data/Final_tiles_1kb_sequences_to_test.fa
      		seq_name=$(basename $seq)
          bin/my_bsub_gridengine -d "$ID_main" -n predict_peaks_${JOB_ID} -o $OUTDIR/log_predictions -m 10 -T '1:00:00' "$Script_dir/Predict_CNN_model_from_fasta.py -s $seq -m $OUTDIR/Model -o $OUTDIR"
          
          # Predict with ATAC-seq model for final test set
      		OUTDIR_ATAC=Enhancer_activity_models/Models/Results_${fold}_${var}_rep${rep}_ATAC_predictions
        	seq=Enhancer_activity_models/Data/Final_tiles_1kb_sequences_to_test.fa
      		seq_name=$(basename $seq)
          bin/my_bsub_gridengine -n predict_peaks_${JOB_ID} -o $OUTDIR_ATAC/log_predictions -m 10 -T '1:00:00' "$Script_dir/Predict_CNN_model_from_fasta.py -s $seq -m $arch -o $OUTDIR_ATAC"
          
      
      		# further analyses only for specific model replicates
      		if [ "$rep" == "1" ]; then
      
      			# Contr scores for merged peaks
            mkdir -p $OUTDIR/log_DeepExplainer
            bin/my_bsub_gridengine -d "$ID_main" -m 40 -T '10:00:00' -P g -G "gpu:1" -o $OUTDIR/log_DeepExplainer -n DeepExplainer_${JOB_ID} "$Script_dir/run_DeepSHAP_DeepExplainer.py -m $OUTDIR/Model -s $seq -b dinuc_shuffle -l -2" > $OUTDIR/log_DeepExplainer/msg.contr_scores_${seq_name}.tmp
      
      			# Convert contr scores to R object
      			ID_contrscores=$(paste $OUTDIR/log_DeepExplainer/msg.contr_scores_${seq_name}.tmp | grep Submitted | awk '{print $4}') # contrscores usually take longer than predictions above, that are also needed
      			bin/my_bsub_gridengine -d "$ID_contrscores" -m 40 -T '08:00:00' -o $OUTDIR/log_DeepExplainer -n DeepExplainer_convert_${JOB_ID} "$Script_dir/Load_and_average_contr_scores.R $var $OUTDIR/Model $seq $OUTDIR/${seq_name}_predictions_Model.txt"
      
      
      			
      			### test initialising with model (e.g. salivarygland) of other tissue for comparison
      			arch=Accessibility_models/Models/Results_${fold}_salivarygland_DeepSTARR2_rep${rep}/Model
      			OUTDIR=Enhancer_activity_models/Models/Results_${fold}_${var}_rep${rep}_init_salivarygland
      			mkdir -p $OUTDIR/log_training
      			bin/my_bsub_gridengine -P g -G "gpu:1" -m 50 -T '01:00:00' -o $OUTDIR/log_training -n Training_${JOB_ID}_init_salivarygland "$Script_dir/Train_transfer_learning_model.py -i Data/$fold -v $var -a $arch -o $OUTDIR/Model" > $OUTDIR/log_training/msg.model_training.tmp
      
      			# Predict for merged peaks
        		ID_main=$(paste $OUTDIR/log_training/msg.model_training.tmp | grep Submitted | awk '{print $4}')
        		seq=Accessibility_models/Data/Sequences_merged_peaks_1kb.fa
        		seq_name=$(basename $seq)
            bin/my_bsub_gridengine -d "$ID_main" -n predict_peaks_${JOB_ID} -o $OUTDIR/log_predictions -m 10 -T '1:00:00' "$Script_dir/Predict_CNN_model_from_fasta.py -s $seq -m $OUTDIR/Model -o $OUTDIR"
            
            # Predict for final test set
        		seq=Enhancer_activity_models/Data/Final_tiles_1kb_sequences_to_test.fa
        		seq_name=$(basename $seq)
            bin/my_bsub_gridengine -d "$ID_main" -n predict_peaks_${JOB_ID} -o $OUTDIR/log_predictions -m 10 -T '1:00:00' "$Script_dir/Predict_CNN_model_from_fasta.py -s $seq -m $OUTDIR/Model -o $OUTDIR"
            
            
      			### train with random initialization
      			ID_main=$(paste $OUTDIR/log_training/msg.model_training.tmp | grep Submitted | awk '{print $4}')
      			OUTDIR=Enhancer_activity_models/Models/Results_${fold}_${var}_rep${rep}_init_random
      			mkdir -p $OUTDIR/log_training
      			bin/my_bsub_gridengine -P g -G "gpu:1" -m 50 -T '03:00:00' -o $OUTDIR/log_training -n Training_${JOB_ID}_init_random "$Script_dir/Train_random_initi_model.py -i Data/$fold -v $var -o $OUTDIR/Model" > $OUTDIR/log_training/msg.model_training.tmp
      			
      			# Predict for merged peaks
        		seq=Accessibility_models/Data/Sequences_merged_peaks_1kb.fa
        		seq_name=$(basename $seq)
            bin/my_bsub_gridengine -d "$ID_main" -n predict_peaks_${JOB_ID} -o $OUTDIR/log_predictions -m 10 -T '1:00:00' "$Script_dir/Predict_CNN_model_from_fasta.py -s $seq -m $OUTDIR/Model -o $OUTDIR"
            
            # Predict for final test set
        		seq=Enhancer_activity_models/Data/Final_tiles_1kb_sequences_to_test.fa
        		seq_name=$(basename $seq)
            bin/my_bsub_gridengine -d "$ID_main" -n predict_peaks_${JOB_ID} -o $OUTDIR/log_predictions -m 10 -T '1:00:00' "$Script_dir/Predict_CNN_model_from_fasta.py -s $seq -m $OUTDIR/Model -o $OUTDIR"
            
            
        	fi

  	  	done
  	done
done

