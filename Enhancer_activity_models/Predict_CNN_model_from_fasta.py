
### Load arguments

import sys, getopt

def main(argv):
   new_seq = ''
   model_ID = ''
   try:
      opts, args = getopt.getopt(argv,"hs:m:o:",["seq=","model=","outdir="])
   except getopt.GetoptError:
      print('Predict_enh_activity_CNN_model_from_fasta.py -s <fasta seq file> -m <CNN model file> -o <output directory>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Predict_enh_activity_CNN_model_from_fasta.py -s <fasta seq file> -m <CNN model file> -o <output directory>')
         sys.exit()
      elif opt in ("-s", "--seq"):
         new_seq = arg
      elif opt in ("-m", "--model"):
         model_ID = arg
      elif opt in ("-o", "--outdir"):
         outdir = arg
   if new_seq=='': sys.exit("fasta seq file not found")
   if model_ID=='': sys.exit("CNN model file not found")
   if outdir=='': sys.exit("outdir not found")
   print('Input Fasta file is ', new_seq)
   print('Model file is ', model_ID)
   print('Output directory is ', outdir)
   return new_seq, model_ID, outdir

if __name__ == "__main__":
   new_seq, model_ID, outdir = main(sys.argv[1:])




### Load libraries

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, InputLayer, Input, GlobalAvgPool1D, GlobalMaxPooling1D, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import plot_model

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

import sys
sys.path.append('bin/')
from helper import IOHelper, SequenceHelper # from https://github.com/bernardo-de-almeida/Neural_Network_DNA_Demo.git

import random
random.seed(1234)

### check number of GPUs used
import tensorflow as tf
print("\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), "\n")



### Load sequences
print("\nLoading sequences ...\n")
input_fasta = IOHelper.get_fastas_from_file(new_seq, uppercase=True)
print(input_fasta.shape)

# length of first sequence
sequence_length = len(input_fasta.sequence.iloc[0])

# Convert sequence to one hot encoding matrix
seq_matrix = SequenceHelper.do_one_hot_encoding(input_fasta.sequence, sequence_length,
                                                SequenceHelper.parse_alpha_to_seq)

### load model
def load_model(model_path):
    import deeplift
    from tensorflow.keras.models import model_from_json
    keras_model_weights = model_path + '.h5'
    keras_model_json = model_path + '.json'
    keras_model = model_from_json(open(keras_model_json).read())
    keras_model.load_weights(keras_model_weights)
    #keras_model.summary()
    return keras_model, keras_model_weights, keras_model_json

keras_model, keras_model_weights, keras_model_json = load_model(model_ID)

### predict dev and hk activity
print("\nPredicting ...\n")
pred=keras_model.predict(seq_matrix)

out_prediction = input_fasta
out_prediction['Predictions'] = pred
out_prediction.drop('sequence', inplace=True, axis=1) # drop DNA sequence column

### save file
print("\nSaving file ...\n")
import os.path
model_ID_out=os.path.basename(model_ID)
new_seq_out=os.path.basename(new_seq)
out_prediction.to_csv(outdir + "/" + new_seq_out + "_predictions_" + model_ID_out + ".txt", sep="\t", index=False)
