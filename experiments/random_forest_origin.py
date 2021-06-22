from google.colab import drive
from google.colab import files

drive.mount('drive')

import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
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

#X_cwt = load_dataset('drive/My Drive/SCA_diploma/sym3[1,100000]-_by_max_cor_freqs-scales[1:10:1000].mat')
y = np.load('drive/My Drive/SCA_diploma//Y_100k.npy')

tscv = KFold(n_splits=3)

k = 50000
pred_n_bits = 1

data = X_100k[-k:]
X_cwt_test = X_100k[:k]


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
  return list_accuracy
  

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
# rf_lgb = LGBMClassifier(learning_rate=0.05)

# lgb_cv_score, oof_preds = validate_model(rf_lgb, data2, target, pred_n_bits)
# print(lgb_cv_score)


# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform

# logistic = RandomForestClassifier()
# distributions = dict(n_estimators=[100,200,300,400,500],
#                      min_samples_split=[2,3,5,7,10,15],
#                      max_depth=[3,5,7,10,15,20])
# clf = RandomizedSearchCV(logistic, distributions, random_state=0)
# search = clf.fit(data2, target)
# search.best_params_


%%time
rf_lgb = RandomForestClassifier(max_depth= 20, min_samples_split= 5, n_estimators= 300)
fitted_model = MultiOutputClassifier(rf_lgb).fit(data2, target)
test_preds = evaluate_model(fitted_model, X_cwt_test2, test_target, pred_n_bits)


# list_accuracy = []
# for i in range(pred_n_bits):
#     list_accuracy.append(accuracy_score(test_target[:, i], test_preds[:, i]))

# sns.set_style("darkgrid")
# plt.plot(sorted(list_accuracy))
# plt.show()


pd.DataFrame(data=np.array(test_preds).reshape(128,1))


test_preds_df = pd.DataFrame(data=test_preds,columns=['bit'+str(i) for i in range(1,test_preds.shape[1]+1)])
test_preds_df.to_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_rf_origin.csv')
dump(fitted_model, '/content/drive/MyDrive/SCA_diploma/preds and models/rf_origin.joblib')


dump(fitted_model, '/content/drive/MyDrive/SCA_diploma/preds and models/rf_origin.joblib')


def evaluate_model(model, test, target, pred_n_bits):
  mean_accuracy = []
  preds = model.predict(test)

  list_accuracy = []
  for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(target[:, i], preds[:, i]))
  mean_accuracy.append(sum(list_accuracy)/len(list_accuracy))
  print(str(model).split('(')[1].split('=')[1], sum(mean_accuracy)/len(mean_accuracy))
  return list_accuracy
  
  
list_accuracy_rf_origin = evaluate_model(fitted_model, X_cwt_test2, test_target, pred_n_bits)


#X_100k = np.load('drive/My Drive/SCA_diploma/X_100k.npy')

X_cwt = load_dataset('drive/My Drive/SCA_diploma/sym3[1,100000]-_by_max_cor_freqs-scales[1:10:1000].mat')
y = np.load('drive/My Drive/SCA_diploma//Y_100k.npy')

tscv = KFold(n_splits=3)

k = 5000
pred_n_bits = 128

data = X_cwt[-k:]
X_cwt_test = X_cwt[:k]


test_target = y[:k, :pred_n_bits]
target = y[-k:, :pred_n_bits]

data_common = []
for i in range(len(data)):
  tmp = []
  for j in range(len(data[0])):
    if j%6==0:
      tmp.append(data[i][j-1]+data[i][j]+data[i][j-2]+data[i][j-3]+data[i][j-4]+data[i][j-5])
  data_common.append(tmp)
data_common = np.array(data_common)

X_cwt_test_common = []
for i in range(len(X_cwt_test)):
  tmp = []
  for j in range(len(X_cwt_test[0])):
    if j%6==0:
      tmp.append(X_cwt_test[i][j-1]+X_cwt_test[i][j]+X_cwt_test[i][j-2]+X_cwt_test[i][j-3]+X_cwt_test[i][j-4]+X_cwt_test[i][j-5])
  X_cwt_test_common.append(tmp)
X_cwt_test_common = np.array(X_cwt_test_common)

rf_clf = load('/content/drive/MyDrive/SCA_diploma/preds and models/rf.joblib')


list_accuracy_rf_common = evaluate_model(rf_clf, X_cwt_test_common, test_target, pred_n_bits)


def evaluate_model_stacked(model1, model2, test, test_origin, target, pred_n_bits):
  mean_accuracy = []
  preds1 = model1.predict_proba(test)
  preds2 = model2.predict_proba(test_origin)

  list_accuracy = []
  for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(target[:, i], (0.3*pd.DataFrame(preds1[i][:,1])+0.7*pd.DataFrame(preds2[i][:,1])).round().astype(int).iloc[:,0].values))
  mean_accuracy.append(sum(list_accuracy)/len(list_accuracy))
  print(sum(mean_accuracy)/len(mean_accuracy))
  return list_accuracy
  
  
list_accuracy_lr_stacked = evaluate_model_stacked(rf_clf, fitted_model, X_cwt_test_common, X_cwt_test2, test_target, pred_n_bits)


rf_df = pd.DataFrame(list_accuracy_lr_stacked,columns=['stacked'])


#cross validation for linear and trees models
tscv = KFold(n_splits=5)
fold_scores = []
oof_preds = np.zeros((X_cwt_test2.shape[0], pred_n_bits))
for n_fold, (train_idx, val_idx) in enumerate(tscv.split(X_cwt_test2)):
    x_val, y_val = X_cwt_test2[val_idx], test_target[val_idx]
    
    clf = fitted_model
    oof_preds[val_idx] = clf.predict(x_val)

    mean_accuracy = []
    for i in range(pred_n_bits):
      mean_accuracy.append(accuracy_score(test_target[val_idx, i], oof_preds[val_idx, i]))
    fold_scores.append(sum(mean_accuracy)/len(mean_accuracy))
print('fold scores =',fold_scores)
print(clf)
print('std =', np.array(fold_scores).std())
print('mean =', np.array(fold_scores).mean())


rf_df.to_csv('/content/drive/MyDrive/SCA_diploma/preds and models/rf_df.csv')


def bias(arr):
  return max(abs(arr.mean()-max(arr)),abs(arr.mean()-min(arr)))
def validate_model(model, test,test_target, pred_n_bits):
  tscv = KFold(n_splits=5)
  fold_scores = []
  oof_preds = np.zeros((test.shape[0], pred_n_bits))
  for n_fold, (train_idx, val_idx) in enumerate(tscv.split(test)):
      x_val, y_val = test[val_idx], test_target[val_idx]
      
      clf = model
      oof_preds[val_idx] = clf.predict(x_val)

      mean_accuracy = []
      for i in range(pred_n_bits):
        mean_accuracy.append(accuracy_score(test_target[val_idx, i], oof_preds[val_idx, i]))
      fold_scores.append(sum(mean_accuracy)/len(mean_accuracy))
  print('fold scores =',fold_scores)
  print(clf)
  print('std =', np.array(fold_scores).std())
  print('bias =', bias(np.array(fold_scores)))
  print('mean =', np.array(fold_scores).mean())
  

validate_model(rf_clf, X_cwt_test_common, test_target, pred_n_bits)

validate_model(fitted_model, X_cwt_test2, test_target, pred_n_bits)


def validate_model_stacked(model1, model2, test, test_origin, target, pred_n_bits):

  kf = KFold(n_splits=5)
  fold_scores = []
  oof_preds1 = np.zeros((pred_n_bits,test.shape[0]))
  oof_preds2 = np.zeros((pred_n_bits,test_origin.shape[0]))
  for n_fold, (train_idx, val_idx) in enumerate(kf.split(test_origin)):
      x_val, y_val = test[val_idx], test[val_idx]
      x_val_orig, y_val_orig = test_origin[val_idx], test_origin[val_idx]
      oof_preds1[:,val_idx] = np.array(model1.predict_proba(x_val))[:,:,1]
      oof_preds2[:,val_idx] = np.array(model2.predict_proba(x_val_orig))[:,:,1]
      mean_accuracy = []
      for i in range(pred_n_bits):
        mean_accuracy.append(accuracy_score(target[val_idx, i], (0.4*oof_preds1[i][val_idx]+0.6*oof_preds2[i][val_idx]).round()))
      
      fold_scores.append(sum(mean_accuracy)/len(mean_accuracy))
  print('fold scores =',fold_scores)
  print(model1,model2)
  print('std =', np.array(fold_scores).std())
  print('mean =', np.array(fold_scores).mean())
  print('bias =', bias(np.array(fold_scores)))
  print('-------------------------------------------------------------------------------------------------------------')
  
  
def validate_model_stacked_v3(model1, model2, test, test_origin, target, pred_n_bits):
  diff_scores = []
  p_i = np.zeros(2)
  n_splits = 8
  kf = KFold(n_splits=n_splits)
  scores_1 = []
  scores_2 = [] 
  fold_scores1 = []
  fold_scores2 = []
  oof_preds1 = np.zeros((pred_n_bits,test.shape[0]))
  oof_preds2 = np.zeros((pred_n_bits,test_origin.shape[0]))
  for n_fold, (train_idx, val_idx) in enumerate(kf.split(test_origin)):
      x_val, y_val = test[val_idx], test[val_idx]
      x_val_orig, y_val_orig = test_origin[val_idx], test_origin[val_idx]
      oof_preds1[:,val_idx] = np.array(model1.predict_proba(x_val))[:,:,1]
      oof_preds2[:,val_idx] = np.array(model2.predict_proba(x_val_orig))[:,:,1]
      mean_accuracy1 = []
      mean_accuracy2 = []
      for i in range(pred_n_bits):
        mean_accuracy1.append(accuracy_score(target[val_idx, i], (0.4*oof_preds1[i][val_idx]+0.6*oof_preds2[i][val_idx]).round()))
        mean_accuracy2.append(accuracy_score(target[val_idx, i], (oof_preds2[i][val_idx]).round()))
      score_1 = sum(mean_accuracy1)/len(mean_accuracy1)
      score_2 = sum(mean_accuracy2)/len(mean_accuracy2) 
      scores_1.append(score_1)
      scores_2.append(score_2)
      diff_scores.append(score_1 - score_2)
      fold_scores1.append(sum(mean_accuracy1)/len(mean_accuracy1))
      fold_scores2.append(sum(mean_accuracy2)/len(mean_accuracy2))
  centered_diff = np.array(diff_scores) - np.mean(diff_scores)
  tt = np.mean(diff_scores) * (n_splits ** .5) / (np.sqrt(np.sum(centered_diff ** 2) / (n_splits - 1)))
  pval = stats.t.sf(np.abs(tt), n_splits-1)*2 
  print('t value =', tt,'p value =',pval)
  print('fold scores stacked models =',fold_scores1)
  print('fold scores origin model =',fold_scores2)
  print(model1,model2)
  # print('std =', np.array(fold_scores1).std())
  # print('mean =', np.array(fold_scores1).mean())
  # print('bias =', bias(np.array(fold_scores1)))
  print('-------------------------------------------------------------------------------------------------------------')
  
  
from scipy import stats
validate_model_stacked_v3(rf_clf, fitted_model, X_cwt_test_common, X_cwt_test2, test_target, pred_n_bits)


validate_model_stacked(rf_clf, fitted_model, X_cwt_test_common, X_cwt_test2, test_target, pred_n_bits)


oof_preds1 = np.zeros((pred_n_bits,X_cwt_test_common.shape[0]))
oof_preds1[:,[0,1,2]] = np.array(rf_clf.predict_proba(X_cwt_test_common[[0,1,2]]))[:,:,1]