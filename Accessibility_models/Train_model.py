
#########
### Load arguments
#########

import sys, getopt

def main(argv):
   conv_n = '4'
   dense_n = '2'
   conv_filt = '256'
   filt_size = '7'
   model_ID_output = ''
   try:
      opts, args = getopt.getopt(argv,"hi:v:a:o:")
   except getopt.GetoptError:
      print('Train_model.py -i <fold> -v <output variable> -a <architecture> -o <output model ID>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Train_model.py -i <fold> -v <output variable> -a <architecture> -o <output model ID>')
         sys.exit()
      elif opt in ("-i"):
         fold = arg
      elif opt in ("-v"):
         output = arg
      elif opt in ("-a"):
         architecture = arg
      elif opt in ("-o"):
         model_ID_output = arg
   if fold=='': sys.exit("fold not found")
   if output=='': sys.exit("variable output not found")
   if architecture=='': sys.exit("architecture not found")
   if model_ID_output=='': sys.exit("Output model ID not found")
   print('fold ', fold)
   print('variable output ', output)
   print('Model architecture ', architecture)
   print('Output model ID is ', model_ID_output)
   return fold, output, architecture, model_ID_output

if __name__ == "__main__":
   fold, output, architecture, model_ID_output = main(sys.argv[1:])


#########
### Load libraries
#########

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
# random.seed(1234)

### check number of GPUs used
import tensorflow as tf
print("\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), "\n")



#########
### Load sequences and activity
#########

# function to load sequences and output
def prepare_input(fold, set, output):
    # Convert sequences to one-hot encoding matrix
    file_seq = str(fold + "_sequences_" + set + ".fa")
    input_fasta_data_A = IOHelper.get_fastas_from_file(file_seq, uppercase=True)

    # length of first sequence
    sequence_length = len(input_fasta_data_A.sequence.iloc[0])

    # Convert sequence to one hot encoding matrix
    seq_matrix_A = SequenceHelper.do_one_hot_encoding(input_fasta_data_A.sequence, sequence_length,
                                                      SequenceHelper.parse_alpha_to_seq)
    print(seq_matrix_A.shape)
    
    X = np.nan_to_num(seq_matrix_A) # Replace NaN with zero and infinity with large finite numbers
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    Activity = pd.read_table(fold + "_sequences_activity_" + set + ".txt")
    Y = Activity[output]
    
    print(set)

    return input_fasta_data_A, seq_matrix_A, X_reshaped, Y

print("\nLoad sequences\n")

X_train_sequence, X_train_seq_matrix, X_train, Y_train = prepare_input(fold, "Train", output)
X_valid_sequence, X_valid_seq_matrix, X_valid, Y_valid = prepare_input(fold, "Val", output)
X_test_sequence, X_test_seq_matrix, X_test, Y_test = prepare_input(fold, "Test", output)

### Additional metrics
from scipy.stats import spearmanr
def Spearman(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )

#########
### Load model architecture
#########

print("\nBuild model: " + architecture +"\n")

sys.path.append('Accessibility_models')
from Final_model_architectures import *

if architecture == 'DeepSTARR2':
   get_model=DeepSTARR2

get_model()[0].summary()
get_model()[1] # dictionary

#########
### Model training
#########

print("\nModel training")

def train(selected_model, X_train, Y_train, X_valid, Y_valid, params):

	my_history=selected_model.fit(X_train, Y_train,
                                validation_data=(X_valid, Y_valid),
                                batch_size=params['batch_size'], epochs=params['epochs'],
                                callbacks=[EarlyStopping(patience=params['early_stop'], monitor="val_loss", restore_best_weights=True),
                                History()])

	return selected_model, my_history


# tensorflow variables need to be initialized before calling model.fit()
from keras import backend as K
K.get_session().run(tf.local_variables_initializer())

# Model fit
main_model, main_params = get_model()
main_model, my_history = train(main_model, X_train, Y_train, X_valid, Y_valid, main_params)


#########
### Model evaluation
#########

print("\nEvaluating model ...\n")

from scipy import stats
from sklearn.metrics import mean_squared_error

def summary_statistics(X, Y, set):
    pred = main_model.predict(X, batch_size=main_params['batch_size'])
    print(set + ' MSE = ' + str("{0:0.2f}".format(mean_squared_error(Y, pred.squeeze()))))
    print(set + ' PCC = ' + str("{0:0.2f}".format(stats.pearsonr(Y, pred.squeeze())[0])))
    print(set + ' SCC = ' + str("{0:0.2f}".format(stats.spearmanr(Y, pred.squeeze())[0])))
    
summary_statistics(X_train, Y_train, "train")
summary_statistics(X_valid, Y_valid, "validation")
summary_statistics(X_test, Y_test, "test")

#########
### Save model
#########

print("\nSaving model ...\n")

model_json = main_model.to_json()
with open(model_ID_output + '.json', "w") as json_file:
    json_file.write(model_json)
main_model.save_weights(model_ID_output + '.h5')
