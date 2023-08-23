
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
from tensorflow.keras.layers import Bidirectional, Concatenate, PReLU, TimeDistributed 
from tensorflow.keras import regularizers

### Additional metrics
from scipy.stats import spearmanr
def Spearman(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), 
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )


#########
### DeepSTARR2
#########

params_DeepSTARR2 = {'batch_size': 128,
          'epochs': 100,
          'early_stop': 5,
          'kernel_size1': 7,
          'kernel_size2': 3,
          'kernel_size3': 3,
          'kernel_size4': 3,
          'lr': 0.005,
          'num_filters': 256,
          'num_filters2': 120,
          'num_filters3': 60,
          'num_filters4': 60,
          'n_conv_layer': 4,
          'n_add_layer': 2,
          'dropout_prob': 0.4,
          'dense_neurons1': 64,
          'dense_neurons2': 256,
          'pad':'same',
          'act':'relu'}

def DeepSTARR2(params=params_DeepSTARR2):
    
    dropout_prob = params['dropout_prob']
    
    # body
    input = Input(shape=(1001, 4))
    x = Conv1D(params['num_filters'], kernel_size=params['kernel_size1'],
               padding=params['pad'],
               name='Conv1D_1st')(input)
    x = BatchNormalization()(x)
    x = Activation(params['act'])(x)
    x = MaxPooling1D(3)(x)

    for i in range(1, params['n_conv_layer']):
        x = Conv1D(params['num_filters'+str(i+1)],
                   kernel_size=params['kernel_size'+str(i+1)],
                   padding=params['pad'],
                   name=str('Conv1D_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation(params['act'])(x)
        x = MaxPooling1D(3)(x)
    
    x = Flatten()(x)
    
    # dense layers
    for i in range(0, params['n_add_layer']):
        x = Dense(params['dense_neurons'+str(i+1)],
                  name=str('Dense_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation(params['act'])(x)
        x = Dropout(dropout_prob)(x)
    bottleneck = x
    
    # single output
    outputs = Dense(1, activation='linear', name=str('Dense_output'))(bottleneck)

    model = Model([input], outputs)
    model.compile(Adam(lr=params['lr']),
                  loss='mse', # loss
                  metrics=[Spearman]) # additional track metric

    return model, params

