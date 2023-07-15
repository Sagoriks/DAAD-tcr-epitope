import numpy as np
import pandas as pd
dataset_pos = pd.read_pickle('pos_beta_pair_embeds_2048_full.pkl')
dataset_neg = pd.read_pickle('neg_beta_pair_embeds_2048_10x_1.pkl')
dataset = pd.concat(dataset_pos , dataset_neg)
dataset['tcr_epi_embeds'] = dataset['tcr_epi_embeds'].squeeze()
X = dataset['tcr_epi_embeds']
y = dataset['binding']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
X_train_cv = np.array((X).tolist())
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators=300, max_features=15, bootstrap=False, random_state=42)
scores_RF = cross_val_score(clf_RF, X_train_cv, y, cv=10)
clf_RF.fit(np.array((X).tolist()), y)
y_pred_RF = clf_RF.predict(np.array((X_test).tolist()))
from sklearn.metrics import accuracy_score
acc_RF = accuracy_score(y_test, y_pred_RF)
from sklearn.metrics import precision_score
pre_RF = precision_score(y_test, y_pred_RF)
from sklearn.metrics import recall_score
rec_RF = recall_score(y_test, y_pred_RF)
from sklearn.metrics import f1_score
f1_RF = f1_score(y_test, y_pred_RF)
from sklearn.metrics import confusion_matrix
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, y_pred_RF).ravel()
specificity_RF = tn_rf / (tn_rf+fp_rf)
from sklearn.metrics import matthews_corrcoef
mat_RF = matthews_corrcoef(y_test, y_pred_RF)
from sklearn import metrics
fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(y_test, y_pred_RF)
auc_RF = metrics.auc(fpr_rf, tpr_rf)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_RF)
print(cm)
file2 = open('RF_metrics_1_1_epitcr.txt', 'w')
file2.write("%0.2f accuracy with a standard deviation of %0.2f" % (scores_RF.mean(), scores_RF.std()) + '\n')
file2.write("CV Scores" + str(scores_RF) + '\n')
file2.write("Accuracy = " + str(acc_RF) + '\n')
file2.write("Precision = " + str(pre_RF) + '\n')
file2.write("Recall = " + str(rec_RF) + '\n')
file2.write("tn_rf = " + str(tn_rf) + '\n')
file2.write("fp_rf = " + str(fp_rf) + '\n')
file2.write("fn_rf = " + str(fn_rf) + '\n')
file2.write("tp_rf = " + str(tp_rf) + '\n')
file2.write("Confusion_matrix = " + str(cm) + '\n')
file2.write("F1_score = " + str(f1_RF) + '\n')
file2.write("specificity = " + str(specificity_RF) + '\n')
file2.write("Mathew_coorelation = " + str(mat_RF) + '\n')
file2.write("AUC = " + str(auc_RF) + '\n')
file2.close()

