import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt                       # Plotagem de gráficos
from sklearn import metrics                           # Metricas para calcular accuracy score
from sklearn.metrics import RocCurveDisplay



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

with open('dados_numericos/saida.csv', 'r') as csv:
    samples = pd.read_csv(csv)

print(samples.head(15))

X = samples[['Eccentricity', 'Convexicity', 'Aspect Ratio', 'RatioHilumStarch']].values
y = samples['Class'].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"Accuracy {accuracy_score(y_test, y_pred)}")


"""gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(f"Accuracy {accuracy_score(y_test, y_pred)}")
"""

# fine tuning
"""k_range = list(range(1, 31))
#param_grid = dict(n_neighbors=k_range)
param_grid = [{
    'weights': ['uniform', 'distance'], 
    'n_neighbors': k_range,
    'metric' : ['euclidean', 'manhattan']
    }]

grid = GridSearchCV(
    knn, 
    param_grid, 
    cv=10, 
    scoring='accuracy', 
    return_train_score=False,
    verbose=1,
    n_jobs = -1
)

grid_search=grid.fit(X_train, y_train)
print(grid.best_score_)
print("Best Paramers", grid_search.best_params_)"""

"""import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knn.classes_)
disp.plot(cmap="Blues")

import tikzplotlib
tikzplotlib.save("confusion.tex")"""

#import tikzplotlib

#AUC Curve
lr_probs = knn.predict_proba(X_test)
lr_probs = lr_probs[:, 1]

print(lr_probs)

ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)

lr_auc = roc_auc_score(y_test, lr_probs)

y_test = [0 if each_element == "araruta" else 1 for each_element in y_test]


ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
#plt.show()

import tikzplotlib
tikzplotlib.save("aoc.tex")

"""
# y_test = [0 if each_element == "mandioca" else 1 for each_element in y_test]
# y_pred = [0 if each_element == "mandioca" else 1 for each_element in y_pred]

from sklearn.metrics import roc_curve

y = [0 if each_element == "araruta" else 1 for each_element in y_test]
pred = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y, pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')
display.plot()"""

auc = metrics.auc(lr_fpr, lr_tpr)
print(auc)

plt.show()
knnPickle = open('modelos/knnpickle_file', 'wb') 
pickle.dump(knn, knnPickle)                      
