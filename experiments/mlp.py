from google.colab import drive
from google.colab import files

drive.mount('drive')


import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
import gc
warnings.filterwarnings('ignore')


def load_dataset(name):
  f = h5py.File(name, 'r')
  X = f['traces'][()].T
  return X
  
  
#X_100k = np.load('drive/My Drive/SCA_diploma/X_100k.npy')

X_cwt = load_dataset('drive/My Drive/sym3[1,100000]-_by_max_cor_freqs-scales[1:10:1000].mat')
y = np.load('drive/My Drive/Y_100k.npy')

tscv = KFold(n_splits=3)

k = 50000
pred_n_bits = 128

data = X_cwt[-k:]
target = y[-k:, :pred_n_bits]

X_cwt_test = X_cwt[:k]
test_target = y[:k, :pred_n_bits]


def validate_model(model, data, target, pred_n_bits):
  fold_scores = []
  oof_preds = np.zeros((data.shape[0], pred_n_bits))
  for n_fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
      x_train, y_train = data[train_idx], target[train_idx]
      x_val, y_val = data[val_idx], target[val_idx]
      
      clf = MultiOutputClassifier(model).fit(x_train, y_train)
      oof_preds[val_idx] = clf.predict(x_val)

      mean_accuracy = []
      for i in range(pred_n_bits):
        mean_accuracy.append(accuracy_score(target[val_idx, i], oof_preds[val_idx, i]))
      fold_scores.append(sum(mean_accuracy)/len(mean_accuracy))
      print(fold_scores)
  return sum(fold_scores)/len(fold_scores), oof_preds

def evaluate_model(model, test, target, pred_n_bits):
  mean_accuracy = []
  preds = model.predict(test)
  preds = pd.DataFrame(preds).round().astype(int).values
  list_accuracy = []
  for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(target[:, i], preds[:, i]))
  mean_accuracy.append(sum(list_accuracy)/len(list_accuracy))
  print(sum(mean_accuracy)/len(mean_accuracy))
  return preds
  
  
%%time
data2 = []
for i in range(len(data)):
  tmp = []
  for j in range(len(data[0])):
    if j%6==0:
      tmp.append(data[i][j-1]+data[i][j]+data[i][j-2]+data[i][j-3]+data[i][j-4]+data[i][j-5])
  data2.append(tmp)

X_cwt_test2 = []
for i in range(len(X_cwt_test)):
  tmp = []
  for j in range(len(X_cwt_test[0])):
    if j%6==0:
      tmp.append(X_cwt_test[i][j-1]+X_cwt_test[i][j]+X_cwt_test[i][j-2]+X_cwt_test[i][j-3]+X_cwt_test[i][j-4]+X_cwt_test[i][j-5])
  X_cwt_test2.append(tmp)

data2 = np.asarray(data2)
target = np.asarray(target)


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout
from keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=15)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

model = keras.Sequential()

model.add(Dense(130, input_dim=543, activation='relu'))

model.add(Dense(128, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')

model.fit(data2, target, epochs=500, batch_size=32, callbacks=[early_stopping], validation_split=0.3)

test_preds = evaluate_model(model, X_cwt_test2, test_target, pred_n_bits)


test_preds_df = pd.DataFrame(data=test_preds,columns=['bit'+str(i) for i in range(1,test_preds.shape[1]+1)])
test_preds_df.to_csv('/content/drive/MyDrive/test_preds_mlp.csv')


from keras.models import model_from_json
model_json = model.to_json()
with open("/content/drive/MyDrive/mlp.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/content/drive/MyDrive/mlp.h5")
print("Saved model to disk")
 
# load json and create model
json_file = open('/content/drive/MyDrive/mlp.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/MyDrive/mlp.h5")
print("Loaded model from disk")


test_preds = evaluate_model(loaded_model, X_cwt_test2, test_target, pred_n_bits)