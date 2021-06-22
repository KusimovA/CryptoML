from google.colab import drive
from google.colab import files

drive.mount('drive')


import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from joblib import dump, load
import warnings
import gc
warnings.filterwarnings('ignore')


def load_dataset(name):
  f = h5py.File(name, 'r')
  X = f['traces'][()].T
  return X
  
  
X_100k = np.load('drive/My Drive/SCA_diploma/X_100k.npy')

X_cwt = load_dataset('drive/My Drive/SCA_diploma/sym3[1,100000]-_by_max_cor_freqs-scales[1:10:1000].mat')
y = np.load('drive/My Drive/SCA_diploma//Y_100k.npy')

tscv = KFold(n_splits=3)

k = 5000
pred_n_bits = 1

data = X_cwt[-k:]
X_cwt_test = X_cwt[:k]


test_target = y[:k, :pred_n_bits]
target = y[-k:, :pred_n_bits]


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

  list_accuracy = []
  for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(target[:, i], preds[:, i]))
  mean_accuracy.append(sum(list_accuracy)/len(list_accuracy))
  print(sum(mean_accuracy)/len(mean_accuracy))
  return preds
  
  
data2 = []
for i in range(len(data)):
  tmp = []
  for j in range(len(data[0])):
    if j%6==0:
      tmp.append(data[i][j-1]+data[i][j]+data[i][j-2]+data[i][j-3]+data[i][j-4]+data[i][j-5])
  data2.append(tmp)
data2 = np.array(data2)

X_cwt_test2 = []
for i in range(len(X_cwt_test)):
  tmp = []
  for j in range(len(X_cwt_test[0])):
    if j%6==0:
      tmp.append(X_cwt_test[i][j-1]+X_cwt_test[i][j]+X_cwt_test[i][j-2]+X_cwt_test[i][j-3]+X_cwt_test[i][j-4]+X_cwt_test[i][j-5])
  X_cwt_test2.append(tmp)
X_cwt_test2 = np.array(X_cwt_test2)


# %%time
# model_lgb = LGBMClassifier(learning_rate=0.05)

# lgb_cv_score, oof_preds = validate_model(model_lgb, data2, target, pred_n_bits)
# print(lgb_cv_score)


# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform

# logistic = LGBMClassifier()
# distributions = dict(learning_rate=[0.01, 0.05, 0.1],
#                      n_estimators=[100,200,300,400,500],
#                      min_child_samples=[10,20,30,50,100,150],
#                      num_leaves=[5,10,15,20,25,30,50])
# clf = RandomizedSearchCV(logistic, distributions, random_state=0)
# search = clf.fit(data2, target)
# search.best_params_


%%time

model_lgb = LGBMClassifier(learning_rate= 0.05,
 min_child_samples = 30,
 n_estimators = 500,
 num_leaves = 20)
fitted_model = MultiOutputClassifier(model_lgb).fit(data2, target)
test_preds = evaluate_model(fitted_model, X_cwt_test2, test_target, pred_n_bits)


list_accuracy = []
for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(test_target[:, i], test_preds[:, i]))

sns.set_style("darkgrid")
plt.plot(sorted(list_accuracy))
plt.show()


test_preds_df = pd.DataFrame(data=test_preds,columns=['bit'+str(i) for i in range(1,test_preds.shape[1]+1)])
test_preds_df.to_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_lgb.csv')
dump(fitted_model, '/content/drive/MyDrive/SCA_diploma/preds and models/lgb.joblib')