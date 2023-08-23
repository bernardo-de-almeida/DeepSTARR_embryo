
### Run main model across folds and output variables

# models at Accessibility_models/Final_model_architectures.py

mkdir Accessibility_models/Models/

fold_list="fold01,fold02,fold03,fold04,fold05,fold06,fold07,fold08,fold09,fold10"
celltype_IDs="amnioserosa,brain,CNS,epidermis,fatbody,glia,hemocytes,malpighiantube,midgut,pharynx,plasmatocytes,PNS,salivarygland,somaticmuscle,trachea,unknown,ventralmidline,visceralmuscle"

arch=DeepSTARR2

for fold in ${fold_list//,/ }; do
	for var in ${celltype_IDs//,/ }; do
  	  	for rep in 1 2 3; do
  		  
    		  Accessibility_models/Train_DeepSTARR_and_interpretation.sh \
    		  		-d Accessibility_models/Data/$fold -f Accessibility_models/Data/All_sequences_All.fa \
    		  		-a $arch -v $var \
    		  		-o Accessibility_models/Models/Results_${fold}_${var}_${arch}_rep${rep} \
    		  		-c 1 -p Accessibility_models/Data/Sequences_merged_peaks_1kb.fa,Accessibility_models/Data/Sequences_${var}_peaks_1kb.fa \
    		  		-m 1
    		  		
  	  	done
  	done
done