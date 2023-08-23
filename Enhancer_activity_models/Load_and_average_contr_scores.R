#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

require(rhdf5)
require(seqinr)

v=args[1] #"CNS_log2"
contr_h5=args[2] #"Results_fold03_CNS_DeepSTARR2/Model"
seq=args[3] #"Sequences_CNS_peaks_1kb.fa"
pred_file=args[4] #"Results_fold03_CNS_DeepSTARR2/Sequences_all_peaks_1kb.fa_predictions_Model.txt"

# load sequences and predictions
Peaks <- read.fasta(seq, as.string = T, forceDNAtolower = F)
Predictions <- read.delim(pred_file)

# load scores
mydata <- h5read(paste0(contr_h5, "_", basename(seq), "_dinuc_shuffle_deepSHAP_DeepExplainer_importance_scores.h5"), paste0("contrib_scores/", "class"))
Peaks_contr_scores <- list()
for(i in 1:length(Peaks)){
  out <- mydata[,,i]
  rownames(out) <- c("A", "C", "G" ,"T")
  out_name <- paste0(names(Peaks)[i],
                     "_pred", round(Predictions$Predictions[grep(names(Peaks)[i], Predictions$location)],1))
  Peaks_contr_scores[[out_name]] <- out
}

saveRDS(Peaks_contr_scores, file=paste0(contr_h5, "_", basename(seq), "_dinuc_shuffle_deepSHAP_DeepExplainer_importance_scores.rds"))
