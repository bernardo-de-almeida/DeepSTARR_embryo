
#########
### Load arguments
#########

import sys, getopt

def main(argv):
   model_ID_output = ''
   try:
      opts, args = getopt.getopt(argv,"hi:v:a:o:")
   except getopt.GetoptError:
      print('Train_transfer_learning_model.py -i <fold> -v <output variable> -a <architecture> -o <output model ID>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Train_transfer_learning_model.py -i <fold> -v <output variable> -a <architecture> -o <output model ID>')
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
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, InputLayer, Input, GlobalAvgPool1D, GlobalMaxPooling1D, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import plot_model

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
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
### Plotting functions
#########

import seaborn as sns
import matplotlib.pyplot as plt

def plot_training_history(metric, metric_name):
   fig = plt.figure(figsize=(6, 4))
   plt.plot(my_history.history[metric])
   plt.plot(my_history.history['val_' + metric])
   plt.ylabel(metric_name)
   plt.xlabel('Epoch')
   # plt.ylim(ymin=0)
   plt.legend(['Train', 'Val'], loc='upper left')
   # plt.show()
   return fig

def plot_density(Y_pred, Y_test):
   fig = plt.figure(figsize=(6, 4))
   sns.kdeplot(Y_pred, hue=Y_test, palette="cool", label="Category")
   plt.xlabel("Predictor")
   plt.ylabel("Density")
   return fig

#########
### Load sequences and activity
#########

# function to load sequences and output
def prepare_input(fold, set, output, remove_no_ov=True, balanced_negative=True):
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

    # output
    Activity = pd.read_table(fold + "_sequences_activity_" + set + ".txt")
    
    # Filter for active fragment tiles that overlap peaks - cleaner positive set
    if remove_no_ov:
        IDs = ((Activity[output] == "Active") & (Activity[str(output + "_overlap")] == False))
        Activity = Activity[~IDs]

        input_fasta_data_A = input_fasta_data_A[~IDs]
        seq_matrix_A = seq_matrix_A[~IDs,]
        X_reshaped = X_reshaped[~IDs,]
    
    # select a unique tile per region (or a few), in case I want to remove multiple tiles per negative region to balance the datasets
    if balanced_negative:
        # Set the Main_tile column to "No" for all rows
        Activity["Main_tile"] = "No"

        # Group the data by the ID column and apply a function to each group
        def select_random_rows(group):
            # If the group has fewer than 5 rows, select all rows
            n = 5
            if len(group) < n:
                tmp = group
            else:
                # Otherwise, select 5 random rows from the group
                tmp = group.sample(n=n)
            # Set the Main_tile column to "Yes" for the selected rows
            Activity.loc[tmp.index, "Main_tile"] = "Yes"

        Activity.groupby("ID").apply(select_random_rows)
        IDs2 = (Activity["Main_tile"] == "No") & (Activity[output] == "Inactive")
        Activity = Activity[~IDs2]

        input_fasta_data_A = input_fasta_data_A[~IDs2]
        seq_matrix_A = seq_matrix_A[~IDs2,]
        X_reshaped = X_reshaped[~IDs2,]

        
    Y = Activity[output]
    Y = pd.get_dummies(Y)["Active"]
    
    # Print the frequency of each level in the column
    print(Y.value_counts())
    
    print(set)

    return input_fasta_data_A, seq_matrix_A, X_reshaped, Y

print("\nLoad sequences\n")

X_train_sequence, X_train_seq_matrix, X_train, Y_train = prepare_input(fold, "Train", output)
X_valid_sequence, X_valid_seq_matrix, X_valid, Y_valid = prepare_input(fold, "Val", output)
X_test_sequence, X_test_seq_matrix, X_test, Y_test = prepare_input(fold, "Test", output)

### Additional metrics
def positive_predictive_value_at_cutoff(cutoff=0.5):
  def PPV(y_true, y_pred):
    # Round the predicted probabilities to 0 or 1 using the specified cutoff
    y_pred = tf.cast(y_pred > cutoff, tf.float32)
    # Calculate the true positive rate (sensitivity)
    tp = tf.reduce_sum(y_true * y_pred)
    # Calculate the positive predictive value (PPV)
    ppv = tp / tf.reduce_sum(y_pred)
    return ppv
  return PPV

#########
### Load trained model and change output layer
#########

print("\nInit model: " + architecture +"\n")

def load_model(model_path):
    import deeplift
    from tensorflow.keras.models import model_from_json
    keras_model_weights = model_path + '.h5'
    keras_model_json = model_path + '.json'
    keras_model = model_from_json(open(keras_model_json).read())
    keras_model.load_weights(keras_model_weights)
    #keras_model.summary()
    return keras_model, keras_model_weights, keras_model_json

keras_model, keras_model_weights, keras_model_json = load_model(architecture)

### Get original model, remove last dense layer
DeepSTARR_bottleneck=Model(keras_model.input, keras_model.layers[-2].output)

# check architecture and which layers are trainable
for i,layer in enumerate(DeepSTARR_bottleneck.layers):
    print(i,layer.name, layer.trainable)

# Add binary layer¶
params = {'batch_size': 128,
          'epochs': 200,
          'early_stop': 20,
          'lr': 0.0001, # Steins transfer learning used 0.00001
          'dropout_prob': 0.5}

input = Input(shape=(1001, 4))
x = DeepSTARR_bottleneck(input)
output = Dense(1, name=str('Activity'))(x)
output = Activation('sigmoid')(output)

    
DeepSTARR_tl = Model(input, output)
DeepSTARR_tl.compile(Adam(lr=params['lr']),
                     loss='binary_crossentropy', # loss
                     metrics=['accuracy',
                              positive_predictive_value_at_cutoff(0.5),
                              tf.keras.metrics.FalsePositives(0.5, name="FP"),
                              tf.keras.metrics.FalseNegatives(0.5, name="FN"),
                              tf.keras.metrics.Precision(0.5, name="Prec"),
                              tf.keras.metrics.SpecificityAtSensitivity(0.5, name="SpecAtSens")])

DeepSTARR_tl.summary()

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
from tensorflow.keras import backend as K
K.get_session().run(tf.local_variables_initializer())

# Model fit
main_model, my_history = train(DeepSTARR_tl, X_train, Y_train, X_valid, Y_valid, params)


#########
### Save model
#########

print("\nSaving model ...\n")

model_json = main_model.to_json()
with open(model_ID_output + '.json', "w") as json_file:
    json_file.write(model_json)
main_model.save_weights(model_ID_output + '.h5')

#########
### Model evaluation
#########

print("\nEvaluating model ...\n")

pred = main_model.predict(X_test, batch_size=params['batch_size'])
print("AUC: ", + roc_auc_score(Y_test, pred))
print("AUPRC: ", + average_precision_score(Y_test, pred))
print("Accuracy positive: ", + sum(pred[Y_test == 1] > 0.5) / pred[Y_test == 1].shape[0] * 100, " %")
print("Accuracy negative: ", + sum(pred[Y_test == 0] < 0.5) / pred[Y_test == 0].shape[0] * 100, " %")

# plot training history
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages(model_ID_output + '_evaluation.pdf')
plot1 = plot_training_history("loss", "Loss")
plot2 = plot_training_history("acc", "Accuracy")
plot3 = plot_training_history("PPV", "Positive Predictive Value")
plot4 = plot_training_history("FP", "False positives")
plot5 = plot_training_history("FN", "False negatives")
plot6 = plot_training_history("Prec", "Precision")
plot7 = plot_density(pred.flatten(), Y_test)

pp.savefig(plot1)
pp.savefig(plot2)
pp.savefig(plot3)
pp.savefig(plot4)
pp.savefig(plot5)
pp.savefig(plot6)
pp.savefig(plot7)

pp.close()

print("\nDone ...\n")
