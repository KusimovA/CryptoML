from google.colab import drive
from google.colab import files

drive.mount('drive')


def bias(arr):
  return max(abs(arr.mean()-max(arr)),abs(arr.mean()-min(arr)))
fold_scores= [0.8108331583368336, 0.8143691813663728, 0.8157571223575528, 0.8134602932941342, 0.8142630897382054, 0.8184564433711322, 0.8215846839120696]
print('std =', np.array(fold_scores).std())
print('bias =', bias(np.array(fold_scores)))
print('mean =', np.array(fold_scores).mean())


import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from datetime import datetime
from scipy import stats
from lightgbm import LGBMClassifier
from keras.models import model_from_json
from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler
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

X_cwt = load_dataset('drive/My Drive/SCA_diploma/sym3[1,100000]-_by_max_cor_freqs-scales[1:10:1000].mat')
X_100k = np.load('drive/My Drive/SCA_diploma/X_100k.npy')
y = np.load('drive/My Drive/SCA_diploma//Y_100k.npy')
k = 50000
pred_n_bits = 128

data = X_cwt[-k:]
X_cwt_test = X_cwt[:k]

data_origin = X_100k[-k:]
X_cwt_test_origin = X_100k[:k]


test_target = y[:k, :pred_n_bits]
target = y[-k:, :pred_n_bits]

data_origin2 = []
for i in range(len(data_origin)):
  tmp = []
  for j in range(len(data_origin[0])):
    if j%6==0:
      tmp.append(data_origin[i][j-1]+data_origin[i][j]+data_origin[i][j-2]+data_origin[i][j-3]+data_origin[i][j-4]+data_origin[i][j-5])
  data_origin2.append(tmp)


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

X_cwt_test_origin2 = []
for i in range(len(X_cwt_test_origin)):
  tmp = []
  for j in range(len(X_cwt_test_origin[0])):
    if j%6==0:
      tmp.append(X_cwt_test_origin[i][j-1]+X_cwt_test_origin[i][j]+X_cwt_test_origin[i][j-2]+X_cwt_test_origin[i][j-3]+X_cwt_test_origin[i][j-4]+X_cwt_test_origin[i][j-5])
  X_cwt_test_origin2.append(tmp)
  
  
del X_cwt_test_origin,data,data_origin,X_cwt,X_100k
gc.collect()


test_preds_lgb = pd.read_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_lgb.csv',index_col=0)
logistic_regression_clf = load('/content/drive/MyDrive/SCA_diploma/preds and models/logistic_regression.joblib')

test_preds_logistic_regression = pd.read_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_logistic_regression.csv',index_col=0)
lgb_clf = load('/content/drive/MyDrive/SCA_diploma/preds and models/lgb.joblib')

test_preds_xgb = pd.read_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_xgb.csv',index_col=0)
xgb_clf = load('/content/drive/MyDrive/SCA_diploma/preds and models/xgb.joblib')

test_preds_lgb_origin = pd.read_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_lgb_origin.csv',index_col=0)
logistic_regression_clf_origin = load('/content/drive/MyDrive/SCA_diploma/preds and models/logistic_regression_origin.joblib')

test_preds_logistic_regression_origin = pd.read_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_logistic_regression_origin.csv',index_col=0)
lgb_clf_origin = load('/content/drive/MyDrive/SCA_diploma/preds and models/lgb_origin.joblib')

test_preds_xgb_origin = pd.read_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_xgb_origin.csv',index_col=0)
xgb_clf_origin = load('/content/drive/MyDrive/SCA_diploma/preds and models/xgb_origin.joblib')

test_preds_rf_origin = pd.read_csv('/content/drive/MyDrive/SCA_diploma/preds and models/test_preds_rf_origin.csv',index_col=0)
rf_clf_origin = load('/content/drive/MyDrive/SCA_diploma/preds and models/rf_origin.joblib')

# load json and create model
json_file = open('/content/drive/MyDrive/SCA_diploma/mlp_origin (1).json', 'r')
loaded_model_json = json_file.read()
json_file.close()
mlp_origin = model_from_json(loaded_model_json)
mlp_origin.load_weights("/content/drive/MyDrive/SCA_diploma/mlp_origin (1).h5")
# load json and create model
json_file = open('/content/drive/MyDrive/SCA_diploma/mlp.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
mlp = model_from_json(loaded_model_json)
mlp.load_weights("/content/drive/MyDrive/SCA_diploma/mlp.h5")
# load json and create model
json_file = open('/content/drive/MyDrive/SCA_diploma/cnn_origin.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn_origin = model_from_json(loaded_model_json)
cnn_origin.load_weights("/content/drive/MyDrive/SCA_diploma/cnn_origin.h5")
# load json and create model
json_file = open('/content/drive/MyDrive/SCA_diploma/cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
cnn.load_weights("/content/drive/MyDrive/SCA_diploma/cnn.h5")
# load json and create model
json_file = open('/content/drive/MyDrive/ProposedNN_origin.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ProposedNN_origin = model_from_json(loaded_model_json)
ProposedNN_origin.load_weights("/content/drive/MyDrive/ProposedNN_origin.h5")
# load json and create model
json_file = open('/content/drive/MyDrive/ProposedNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ProposedNN = model_from_json(loaded_model_json)
ProposedNN.load_weights("/content/drive/MyDrive/ProposedNN.h5")


def evaluate_model(model, test, target, pred_n_bits):
  mean_accuracy = []
  preds = model.predict(test)

  list_accuracy = []
  for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(target[:, i], preds[:, i]))
  mean_accuracy.append(sum(list_accuracy)/len(list_accuracy))
  print(str(model).split('(')[1].split('=')[1], sum(mean_accuracy)/len(mean_accuracy))
  return list_accuracy

def evaluate_nn_model(model, test, target, pred_n_bits):
  mean_accuracy = []
  preds = model.predict(test)
  preds = pd.DataFrame(preds).round().astype(int).values
  list_accuracy = []
  for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(target[:, i], preds[:, i]))
  mean_accuracy.append(sum(list_accuracy)/len(list_accuracy))
  print(sum(mean_accuracy)/len(mean_accuracy))
  return list_accuracy

def evaluate_model_stacked(model1, model2, test, test_origin, target, pred_n_bits):
  mean_accuracy = []
  preds1 = model1.predict_proba(test)
  preds2 = model2.predict_proba(test_origin)

  list_accuracy = []
  for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(target[:, i], (0.4*pd.DataFrame(preds1[i][:,1])+0.6*pd.DataFrame(preds2[i][:,1])).round().astype(int).iloc[:,0].values))
  mean_accuracy.append(sum(list_accuracy)/len(list_accuracy))
  print(sum(mean_accuracy)/len(mean_accuracy))
  return list_accuracy

def evaluate_nn_model_stacked(model1, model2, test, test_origin, target, pred_n_bits):
  mean_accuracy = []
  preds1 = model1.predict(test)
  preds2 = model2.predict(test_origin)

  list_accuracy = []
  for i in range(pred_n_bits):
    list_accuracy.append(accuracy_score(target[:, i], (0.4*pd.DataFrame(preds1[:,i])+0.6*pd.DataFrame(preds2[:,i])).round().astype(int).iloc[:,0].values))
  mean_accuracy.append(sum(list_accuracy)/len(list_accuracy))
  print(sum(mean_accuracy)/len(mean_accuracy))
  return list_accuracy
  
  
data2 = np.asarray(data2)
target = np.asarray(target)
X_cwt_test2 = np.asarray(X_cwt_test2)
X_cwt_test_origin2 = np.asarray(X_cwt_test_origin2)

scaler = MinMaxScaler()
scaler.fit(data2)
X_cwt_test2_scaled = scaler.transform(X_cwt_test2)

X_cwt_test22_cnn = X_cwt_test2[:,:].reshape(-1,543,1)
X_cwt_test2_scaled_cnn = X_cwt_test2_scaled[:,:].reshape(-1,543,1)

scaler_origin = MinMaxScaler()
scaler_origin.fit(data_origin2)
X_cwt_test_origin2_scaled = scaler_origin.transform(X_cwt_test_origin2)


rf = pd.read_csv('/content/drive/MyDrive/SCA_diploma/preds and models/rf_df.csv',index_col=0)
list_accuracy_rf = rf.cwt.values
list_accuracy_rf_origin = rf.raw.values
list_accuracy_rf_stacked = rf.stacked.values


list_accuracy_ProposedNN = evaluate_nn_model(ProposedNN, X_cwt_test22_cnn, test_target, pred_n_bits)
list_accuracy_ProposedNN_origin = evaluate_nn_model(ProposedNN_origin, X_cwt_test_origin2_scaled, test_target, pred_n_bits)


list_accuracy_lr = evaluate_model(logistic_regression_clf, X_cwt_test2, test_target, pred_n_bits)
list_accuracy_lgb = evaluate_model(lgb_clf, X_cwt_test2, test_target, pred_n_bits)
list_accuracy_xgb = evaluate_model(xgb_clf, X_cwt_test2, test_target, pred_n_bits)
list_accuracy_mlp = evaluate_nn_model(mlp, X_cwt_test2, test_target, pred_n_bits)
list_accuracy_cnn = evaluate_nn_model(cnn, X_cwt_test22_cnn, test_target, pred_n_bits)

list_accuracy_lr_origin = evaluate_model(logistic_regression_clf_origin, X_cwt_test_origin2, test_target, pred_n_bits)
list_accuracy_lgb_origin = evaluate_model(lgb_clf_origin, X_cwt_test_origin2, test_target, pred_n_bits)
list_accuracy_xgb_origin = evaluate_model(xgb_clf_origin, X_cwt_test_origin2, test_target, pred_n_bits)
list_accuracy_mlp_origin = evaluate_nn_model(mlp_origin, X_cwt_test2_scaled, test_target, pred_n_bits)
list_accuracy_cnn_origin = evaluate_nn_model(cnn_origin, X_cwt_test2_scaled_cnn, test_target, pred_n_bits)


#cross validation for linear and trees models
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
  print('-------------------------------------------------------------------------------------------------')

def validate_nn_model(model, test, test_target, pred_n_bits):
  tscv = KFold(n_splits=5)
  fold_scores = []
  oof_preds = np.zeros((test.shape[0], pred_n_bits))
  for n_fold, (train_idx, val_idx) in enumerate(tscv.split(test)):
      x_val, y_val = test[val_idx], test_target[val_idx]
      
      clf = model
      oof_preds[val_idx] = clf.predict(x_val)

      mean_accuracy = []
      for i in range(pred_n_bits):
        mean_accuracy.append(accuracy_score(test_target[val_idx, i], np.array([round(i) for i in (oof_preds[val_idx, i])])))
      fold_scores.append(sum(mean_accuracy)/len(mean_accuracy))
  print('fold scores =',fold_scores)
  print(clf)
  print('std =', np.array(fold_scores).std())
  print('bias =', bias(np.array(fold_scores)))
  print('mean =', np.array(fold_scores).mean())
  print('-------------------------------------------------------------------------------------------------')
  
  
validate_model(logistic_regression_clf, X_cwt_test2, test_target, pred_n_bits)
validate_model(lgb_clf, X_cwt_test2, test_target, pred_n_bits)
validate_model(xgb_clf, X_cwt_test2, test_target, pred_n_bits)
validate_nn_model(mlp, X_cwt_test2, test_target, pred_n_bits)
validate_nn_model(cnn, X_cwt_test22_cnn, test_target, pred_n_bits)
validate_nn_model(ProposedNN, X_cwt_test22_cnn, test_target, pred_n_bits)

validate_model(logistic_regression_clf_origin, X_cwt_test_origin2, test_target, pred_n_bits)
validate_model(lgb_clf_origin, X_cwt_test_origin2, test_target, pred_n_bits)
validate_model(xgb_clf_origin, X_cwt_test_origin2, test_target, pred_n_bits)
validate_nn_model(mlp_origin, X_cwt_test2_scaled, test_target, pred_n_bits)
validate_nn_model(cnn_origin, X_cwt_test2_scaled_cnn, test_target, pred_n_bits)
validate_nn_model(ProposedNN_origin, X_cwt_test_origin2_scaled, test_target, pred_n_bits)


#cross validation for NN models
tscv = KFold(n_splits=5)
fold_scores = []
oof_preds = np.zeros((X_cwt_test2_scaled_cnn.shape[0], pred_n_bits))
for n_fold, (train_idx, val_idx) in enumerate(tscv.split(X_cwt_test2_scaled_cnn)):
    x_val, y_val = X_cwt_test2_scaled_cnn[val_idx], test_target[val_idx]
    
    clf = cnn_origin
    oof_preds[val_idx] = clf.predict(x_val)

    mean_accuracy = []
    for i in range(pred_n_bits):
      mean_accuracy.append(accuracy_score(test_target[val_idx, i], np.array([round(i) for i in (oof_preds[val_idx, i])])))
    fold_scores.append(sum(mean_accuracy)/len(mean_accuracy))
print('fold scores =',fold_scores)
print(clf)
print('std =', np.array(fold_scores).std())
print('mean =', np.array(fold_scores).mean())


sns.set_style("darkgrid")
plt.plot([list_accuracy_lr[i] for i in list(np.argsort(list_accuracy_lgb))])
plt.plot([list_accuracy_xgb[i] for i in list(np.argsort(list_accuracy_lgb))])

plt.plot([list_accuracy_lr_origin[i] for i in list(np.argsort(list_accuracy_lgb))])
plt.plot([list_accuracy_lgb_origin[i] for i in list(np.argsort(list_accuracy_lgb))])

plt.plot([list_accuracy_xgb_origin[i] for i in list(np.argsort(list_accuracy_lgb))])
plt.plot(sorted(list_accuracy_lgb))
plt.legend(labels=['Logistic Regression','XGBoost','Logistic Regression_origin','LightGBM_origin','XGBoost_origin','LightGBM'])

sns.set(rc={'figure.figsize':(11.7,8.27)})

plt.show()


list_accuracy_lr_stacked = evaluate_model_stacked(logistic_regression_clf, logistic_regression_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
list_accuracy_lgb_stacked = evaluate_model_stacked(lgb_clf, lgb_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
list_accuracy_xgb_stacked = evaluate_model_stacked(xgb_clf, xgb_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
list_accuracy_cnn_stacked = evaluate_nn_model_stacked(cnn, cnn_origin, X_cwt_test22_cnn, X_cwt_test2_scaled_cnn, test_target, pred_n_bits)
list_accuracy_mlp_stacked = evaluate_nn_model_stacked(mlp, mlp_origin, X_cwt_test2, X_cwt_test2_scaled, test_target, pred_n_bits)
list_accuracy_ProposedNN_stacked = evaluate_nn_model_stacked(ProposedNN, ProposedNN_origin, X_cwt_test22_cnn, X_cwt_test_origin2_scaled, test_target, pred_n_bits)


list_accuracy_ProposedNN_stacked = evaluate_nn_model_stacked(ProposedNN, ProposedNN_origin, X_cwt_test22_cnn, X_cwt_test_origin2_scaled, test_target, pred_n_bits)


tscv = KFold(n_splits=5)
fold_scores = []
oof_preds = np.zeros((X_cwt_test2_scaled_cnn.shape[0], pred_n_bits))
for n_fold, (train_idx, val_idx) in enumerate(tscv.split(X_cwt_test2_scaled_cnn)):
    x_val, y_val = X_cwt_test2_scaled_cnn[val_idx], test_target[val_idx]
    
    clf = cnn_origin
    oof_preds[val_idx] = clf.predict(x_val)

    mean_accuracy = []
    for i in range(pred_n_bits):
      mean_accuracy.append(accuracy_score(test_target[val_idx, i], np.array([round(i) for i in (oof_preds[val_idx, i])])))
    fold_scores.append(sum(mean_accuracy)/len(mean_accuracy))
    
    
def validate_model_stacked(model1, model2, test, test_origin, target, pred_n_bits):

  tscv = KFold(n_splits=5)
  fold_scores = []
  oof_preds = np.zeros((test_origin.shape[0], pred_n_bits))
  for n_fold, (train_idx, val_idx) in enumerate(tscv.split(test_origin)):
      x_val, y_val = test[val_idx], test[val_idx]
      x_val_orig, y_val_orig = test_origin[val_idx], test_origin[val_idx]

      clf1 = model1
      clf2 = model2
      oof_preds[val_idx] = 0.4*clf1.predict(x_val)+0.6*clf2.predict(x_val_orig)

      mean_accuracy = []
      for i in range(pred_n_bits):
        mean_accuracy.append(accuracy_score(target[val_idx, i], np.array([round(i) for i in (oof_preds[val_idx, i])])))
      fold_scores.append(sum(mean_accuracy)/len(mean_accuracy))
  print('fold scores =',fold_scores)
  print(clf1,clf2)
  print('std =', np.array(fold_scores).std())
  print('mean =', np.array(fold_scores).mean())
  print('bias =', bias(np.array(fold_scores)))
  print('-------------------------------------------------------------------------------------------------------------')
  
  
def validate_model_stacked_v2(model1, model2, test, test_origin, target, pred_n_bits):

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
  
  
def t_test_8fold_nn(model1, model2, test, test_origin, target, pred_n_bits):
  diff_scores = []
  p_i = np.zeros(2)
  n_splits = 7
  kf = KFold(n_splits=n_splits,random_state=687)
  scores_1 = []
  scores_2 = [] 
  fold_scores1 = []
  fold_scores2 = []
  oof_preds1 = np.zeros((test_origin.shape[0], pred_n_bits))
  oof_preds2 = np.zeros((test_origin.shape[0], pred_n_bits))
  clf1 = model1
  clf2 = model2
  for n_fold, (train_idx, val_idx) in enumerate(kf.split(test_origin)):
      x_val, y_val = test[val_idx], test[val_idx]
      x_val_orig, y_val_orig = test_origin[val_idx], test_origin[val_idx]
      mean_accuracy1 = []
      mean_accuracy2 = []

      oof_preds1[val_idx] = 0.4*clf1.predict(x_val)+0.6*clf2.predict(x_val_orig)
      oof_preds2[val_idx] = clf2.predict(x_val_orig)
      for i in range(pred_n_bits):

        mean_accuracy1.append(accuracy_score(target[val_idx, i], np.array([round(i) for i in (oof_preds1[val_idx, i])])))
        mean_accuracy2.append(accuracy_score(target[val_idx, i], np.array([round(i) for i in (oof_preds2[val_idx, i])])))
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
  
  
def t_test_custom_nn(model1, model2, test, test_origin, target, pred_n_bits):
  diff_scores = []
  p_i = np.zeros(2)
  n_splits = 7
  kf = KFold(n_splits=n_splits,random_state=687)
  scores_1 = []
  scores_2 = [] 
  fold_scores1 = []
  fold_scores2 = []
  oof_preds1 = np.zeros((test_origin.shape[0], pred_n_bits))
  oof_preds2 = np.zeros((test_origin.shape[0], pred_n_bits))
  clf1 = model1
  clf2 = model2
  for n_fold, (train_idx, val_idx) in enumerate(kf.split(test_origin)):
      x_val, y_val = test[val_idx], test[val_idx]
      x_val_orig, y_val_orig = test_origin[val_idx], test_origin[val_idx]
      mean_accuracy1 = []
      mean_accuracy2 = []

      oof_preds1[val_idx] = 0.2*clf1.predict(x_val)+0.8*clf2.predict(x_val_orig)
      oof_preds2[val_idx] = clf2.predict(x_val_orig)
      for i in range(pred_n_bits):

        mean_accuracy1.append(accuracy_score(target[val_idx, i], np.array([round(i) for i in (oof_preds1[val_idx, i])])))
        mean_accuracy2.append(accuracy_score(target[val_idx, i], np.array([round(i) for i in (oof_preds2[val_idx, i])])))
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
  
  
t_test_custom_nn(ProposedNN, ProposedNN_origin, X_cwt_test22_cnn, X_cwt_test_origin2_scaled, test_target, pred_n_bits)


validate_model_stacked_v3(logistic_regression_clf, logistic_regression_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
validate_model_stacked_v3(lgb_clf, lgb_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
validate_model_stacked_v3(xgb_clf, xgb_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
t_test_8fold_nn(cnn, cnn_origin, X_cwt_test22_cnn, X_cwt_test2_scaled_cnn, test_target, pred_n_bits)
t_test_8fold_nn(mlp, mlp_origin, X_cwt_test2, X_cwt_test2_scaled, test_target, pred_n_bits)


t_test_8fold_nn(ProposedNN, ProposedNN_origin, X_cwt_test22_cnn, X_cwt_test_origin2_scaled, test_target, pred_n_bits)


#validate_model_stacked_v2(logistic_regression_clf, logistic_regression_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
#validate_model_stacked_v2(lgb_clf, lgb_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
#validate_model_stacked_v2(xgb_clf, xgb_clf_origin, X_cwt_test2, X_cwt_test_origin2, test_target, pred_n_bits)
validate_model_stacked(cnn, cnn_origin, X_cwt_test22_cnn, X_cwt_test2_scaled_cnn, test_target, pred_n_bits)
#validate_model_stacked(mlp, mlp_origin, X_cwt_test2, X_cwt_test2_scaled, test_target, pred_n_bits)


sns.set_style("whitegrid")

plt.plot([list_accuracy_ProposedNN_origin[i] for i in list(np.argsort(list_accuracy_ProposedNN_stacked))],linewidth=6)

plt.plot(sorted(list_accuracy_ProposedNN_stacked),linewidth=6)
plt.legend(labels=['ProposedNN_without_cwt','ProposedNN_with_cwt'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()


sns.set_style("whitegrid")

plt.plot([list_accuracy_rf_origin[i] for i in list(np.argsort(list_accuracy_rf_stacked))],linewidth=6)

plt.plot(sorted(list_accuracy_rf_stacked),linewidth=6)
plt.legend(labels=['Random_forest_without_cwt','Random_forest_with_cwt'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()


sns.set_style("whitegrid")

plt.plot([list_accuracy_lr_origin[i] for i in list(np.argsort(list_accuracy_lr_stacked))],linewidth=6)

plt.plot(sorted(list_accuracy_lr_stacked),linewidth=6)
plt.legend(labels=['Logistic_Regression_without_cwt','Logistic_Regression_with_cwt'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()


sns.set_style("whitegrid")

plt.plot([list_accuracy_lgb_origin[i] for i in list(np.argsort(list_accuracy_lgb_stacked))],linewidth=6)

plt.plot(sorted(list_accuracy_lgb_stacked),linewidth=6)
plt.legend(labels=['LightGBM_without_cwt','LightGBM_with_cwt'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()


sns.set_style("whitegrid")

plt.plot([list_accuracy_xgb_origin[i] for i in list(np.argsort(list_accuracy_xgb_stacked))],linewidth=6)

plt.plot(sorted(list_accuracy_xgb_stacked),linewidth=6)
plt.legend(labels=['XGBoost_without_cwt','XGBoost_with_cwt'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()


sns.set_style("whitegrid")

plt.plot([list_accuracy_cnn_origin[i] for i in list(np.argsort(list_accuracy_cnn_stacked))],linewidth=6)

plt.plot(sorted(list_accuracy_cnn_stacked),linewidth=6)
plt.legend(labels=['CNN_without_cwt','CNN_with_cwt'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()


sns.set_style("whitegrid")

plt.plot([list_accuracy_mlp_origin[i] for i in list(np.argsort(list_accuracy_mlp_stacked))],linewidth=6)

plt.plot(sorted(list_accuracy_mlp_stacked),linewidth=6)
plt.legend(labels=['MLP_without_cwt','MLP_with_cwt'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()


# sns.set_style("whitegrid")
# plt.title('Comparison of model accuracies with using cwt', size=60)
# plt.plot([list_accuracy_lr[i] for i in list(np.argsort(list_accuracy_cnn))],linewidth=6)
# plt.plot([list_accuracy_xgb[i] for i in list(np.argsort(list_accuracy_cnn))],linewidth=6)

# plt.plot([list_accuracy_lgb[i] for i in list(np.argsort(list_accuracy_cnn))],linewidth=6)
# plt.plot([list_accuracy_mlp[i] for i in list(np.argsort(list_accuracy_cnn))],linewidth=6)
# plt.plot([list_accuracy_ProposedNN[i] for i in list(np.argsort(list_accuracy_cnn))],linewidth=6)

# plt.plot(sorted(list_accuracy_cnn),linewidth=6)
# plt.legend(labels=['Logistic Regression','XGBoost','LightGBM','MLP','Proposed NN','CNN'], prop={'size': 42})
# sns.set(rc={'figure.figsize':(41,34)})
# plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
# plt.ylabel('Accuracy', fontsize=48)
# plt.rc('xtick',labelsize=28)
# plt.rc('ytick',labelsize=28)
# plt.show()


sns.set_style("whitegrid")
plt.ylim(0.4,1)
#plt.xlim(10, 40)
plt.title('Comparison of model accuracies without using cwt', size=60)
plt.plot([list_accuracy_lr_origin[i] for i in list(np.argsort(list_accuracy_ProposedNN_origin))],linewidth=6)
plt.plot([list_accuracy_xgb_origin[i] for i in list(np.argsort(list_accuracy_ProposedNN_origin))],linewidth=6)

plt.plot([list_accuracy_lgb_origin[i] for i in list(np.argsort(list_accuracy_ProposedNN_origin))],linewidth=6)
plt.plot([list_accuracy_mlp_origin[i] for i in list(np.argsort(list_accuracy_ProposedNN_origin))],linewidth=6)
plt.plot([list_accuracy_cnn_origin[i] for i in list(np.argsort(list_accuracy_ProposedNN_origin))],linewidth=6)

plt.plot(sorted(list_accuracy_ProposedNN_origin),linewidth=6)
plt.legend(labels=['Logistic Regression','XGBoost','LightGBM','MLP','CNN','Proposed NN'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()


sns.set_style("whitegrid")
plt.ylim(0.4,1)
#plt.xlim(10, 40)

plt.title('Comparison of combined model accuracies with using cwt', size=60)
plt.plot([list_accuracy_lr_stacked[i] for i in list(np.argsort(list_accuracy_ProposedNN_stacked))],linewidth=6)
plt.plot([list_accuracy_xgb_stacked[i] for i in list(np.argsort(list_accuracy_ProposedNN_stacked))],linewidth=6)

plt.plot([list_accuracy_lgb_stacked[i] for i in list(np.argsort(list_accuracy_ProposedNN_stacked))],linewidth=6)
plt.plot([list_accuracy_mlp_stacked[i] for i in list(np.argsort(list_accuracy_ProposedNN_stacked))],linewidth=6)
plt.plot([list_accuracy_cnn_stacked[i] for i in list(np.argsort(list_accuracy_ProposedNN_stacked))],linewidth=6)

plt.plot(sorted(list_accuracy_ProposedNN_stacked),linewidth=6)
plt.legend(labels=['Logistic Regression','XGBoost','LightGBM','MLP','CNN','Proposed NN'], prop={'size': 42})
sns.set(rc={'figure.figsize':(41,34)})
plt.xlabel('i-th bit of the key sorted by accuracy', fontsize=48)
plt.ylabel('Accuracy', fontsize=48)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.show()
