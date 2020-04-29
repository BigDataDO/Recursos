# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:20:53 2020

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#Data Mining
dataset.head()

#Histogramas
dataset['Age'].plot.hist(title="EDAD",bins=10)

# Arreglando la data vacia
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(X[:, 1:3])
X[:, 1:3] = imp.transform(X[:, 1:3])

#Convertir variable sexo en variables dicotomicas(dummys)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Dividiendo datos en Training set y Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (Normalizacion)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,1:3] = sc.fit_transform(X_train[:,1:3])
X_test[:,1:3] = sc.transform(X_test[:,1:3])

# Fitting/Ajuste de Regresion Logistica a Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicciones de Test set 
y_pred = classifier.predict(X_test)

# Matriz de Confusion 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizando los resultados 
#from mpl_toolkits.mplot3d import Axes3D
X_test = np.asarray(X_test, dtype=np.float)
from matplotlib.colors import ListedColormap
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
 
ax.scatter(X_test[:, 0], X_test[:, 1],X_test[:, 2],  c=y_pred, alpha=0.3, cmap=ListedColormap(['red','blue']))

ax.set_xlabel('Gender')
ax.set_ylabel('Age')
ax.set_zlabel('ESalary')
plt.show()