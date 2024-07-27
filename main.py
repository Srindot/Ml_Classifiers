from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import matplotlib as mp 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml('mnist_784', version=1, as_frame = False)
print(mnist.keys())
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)
print("********************************")
# Select a digit and reshape it for visualization
# Select a digit and reshape it for visualization
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

# Plot the digit
# plt.imshow(some_digit_image, cmap='binary', interpolation='nearest')
# plt.axis("off")
# plt.show()


y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#training a binary classifier 
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# print(sgd_clf.predict([some_digit]))

print("******************************************************************")

# skfolds = StratifiedKFold(n_splits=3)
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

print("******************************************************************")
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confmatrix = confusion_matrix(y_train_5, y_train_pred)
print("Confusion Matrix = ", confmatrix)

print("******************************************************************")

precisionRatio=precision_score(y_train_5, y_train_pred)
recallRatio=recall_score(y_train_5, y_train_pred)
print("precision ration = ", precisionRatio)
print("recallratio = ", recallRatio) 

print("******************************************************************")
f1score = f1_score(y_train_5, y_train_pred)
print("f1score = ", f1score)

print("******************************************************************")

y_scores = sgd_clf.decision_function([some_digit])
print("score Without any threshold = ", y_scores)

threshold = 0
y_some_digits_pred = (y_scores > threshold)
print("score when the threshold is zero = ",y_some_digits_pred)

print("******************************************************************")

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")
print("score using cross validiation  = ",y_scores)

print("******************************************************************")

precisions, recalls, threshold = precision_recall_curve(y_train_5, y_scores)

#definin a function 
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label= "Precision")
    plt.plot(thresholds, recalls[:-1],"g-", label = "Recall")
    plt.title("Precision and Recall Versus the decsion threshold")
    plt.xlabel("Threshold")
    plt.grid()
    plt.show()
#plot_precision_recall_vs_threshold(precisions, recalls, threshold)

print("******************************************************************")
threshold_90_precision = threshold[np.argmax(precisions >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
scoreCustomThreshold = precision_score(y_train_5, y_train_pred_90)
print("scoreCustomThreshold = ",scoreCustomThreshold)
print("******************************************************************")
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1],[0,1], 'k--')
    plt.ylabel("True positive Rate (Recall) ")
    plt.xlabel("False Positive Rate")
    plt.grid()
    plt.show()
# plot_roc_curve(fpr, tpr)

print("******************************************************************")
areaundercurve = roc_auc_score(y_train_5, y_scores)
print("areaundercurve = ", areaundercurve)

print("******************************************************************")
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
# score = proba of positive class
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower right")

print("******************************************************************")
randomforestscore = roc_auc_score(y_train_5, y_scores_forest)
print("randomforestscore = ", randomforestscore)

print("******************************************************************")
#multiclass Classification
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))
some_digit_scores = sgd_clf.decision_function([some_digit])
print("some_digit_scores = ",some_digit_scores)

print("******************************************************************")
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
print(ovo_clf.predict([some_digit]))

print("******************************************************************")
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))

print("******************************************************************")
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

print("******************************************************************")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
score = cross_val_score(sgd_clf,X_train_scaled, y_train, cv = 3, scoring = "accuracy")
print("score of sdg on cross validation = ", score)

print("******************************************************************")
#Error Analysis
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix is ", conf_mx)

#image respresentation 
plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()

print("******************************************************************")
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


print("******************************************************************")

# cl_a, cl_b = 3, 5
# X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# plt.figure(figsize=(8,8))
# plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# plt.show()

print("******************************************************************")
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([some_digit]))
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1score = f1_score(y_multilabel, y_train_knn_pred, average="macro")
print("f1score of multilabel = ",f1score)

print("******************************************************************")

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
# knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[some_index]])
# plot_digit(clean_digit)
print("******************************************************************")