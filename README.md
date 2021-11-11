# Pima-Indians-Diabetes-Support-Vector-Machine-Decision-Tree-Deep-Neural-Network

## Overview

**DATASET LINK** 

https://www.kaggle.com/uciml/pima-indians-diabetes-database

Dataset include health information of female patients (from 21 years old and above) in India.

## Target

The Target is to design Decision Tree, Support Vector Machine and Deep Neural Network to predict whether they would be diabetes or not based on their health information and compare their performance by metrics and Confusion Matrix.

## Features in the dataset

Feature | Description
--------|------------
Pregnancies | Number of times pregnant
Glucose | Plasma glucose concentration over 2 hours in an oral glucose tolerance test
BloodPressure | Diastolic blood pressure (mm Hg)
SkinThickness | Triceps skin fold thickness (mm)
Insulin | 2-Hour serum insulin (mu U/ml)
BMI | Body mass index (weight in kg/(height in m)2)
DiabetesPedigreeFunction | Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
Age | Age (years)
**Outcome** | Label **0 if non-diabetic, 1 if diabetic**

## Data Exploration

* Most of the data features have **continuous value**. The only 2 columns having **categorical value** are Pregnancies and Outcome.

## Data Visualization

### Count plot for Outcome, Pregnancy

*   CountPlot to count labels 0, 1 in the column Outcome
*   CountPlot to count times of Pregnancy
*   Countplot to combine times of Pregnancy and Outcome

![Count plot for Outcome, Pregnancy](https://user-images.githubusercontent.com/70437668/141252482-a5b08e1b-60ce-4b87-86e3-19f2a63f7c09.jpg)

### Box plot

*   BoxPlot for all features in the dataframe (Use parameter orient = 'h' to draw a horizontal boxplot)

![Box plot 1](https://user-images.githubusercontent.com/70437668/141252468-30758f5f-56e6-414e-9e96-4b87e27f4852.jpg)

![Box plot 2](https://user-images.githubusercontent.com/70437668/141252464-b282e445-5e05-44c0-9c33-f293510c42ea.jpg)

![Box plot 3](https://user-images.githubusercontent.com/70437668/141252448-b61599f4-7778-4511-9585-cd3b2d60ccf5.jpg)

![Box plot 4](https://user-images.githubusercontent.com/70437668/141252441-ff7fa857-92ca-400e-8a6c-f9fc59c27a67.jpg)

### Histogram

*   Histogram for all features in the dataframe

![Histogram 1](https://user-images.githubusercontent.com/70437668/141252405-3cacc824-5fe4-4b1f-a752-9c411be577b3.jpg)

![Histogram 2](https://user-images.githubusercontent.com/70437668/141252401-e3c5ec02-4b21-4786-9e81-32727adeabc7.jpg)

### Heatmap for Correlation Matrix

<img src="https://user-images.githubusercontent.com/70437668/141252393-de75c3d5-5617-4c5c-a0d3-3ba12fee42cd.jpg" width=50% height=50%>

## Data Preprocessing

### Observing the Histogram, there are many anamolies in the dataset.

* The features **`Insulin, SkinTickness, BloodPressure, BMI, Glucose`** has many values equal to 0. In these features, value = 0 means that those data points are null (nan / na).

So, I will replace these values by their median value.

Replacing nan values can be done by choosing a Feature with a Categorical type and the highest Correlation, then caculating its Median of data points grouped by that Feature.

However, in this dataset, most of the Features are Continuous types, so I will replace them by Medians of all columns.

For example: I will use the Median valye of all values in BMI to replace its values = 0.

### Draw a Boxplot again after the replacement to see any changes

![Boxplot again](https://user-images.githubusercontent.com/70437668/141252379-3ebac081-786e-4e20-a600-95b2994c3479.jpg)

### Conclusion for Boxplots

* Insulin has no low-outlier values. Values greater than 273 are outliers.

* Glucose has no low-outlier or high-outlier.

* Pregnancies has no low-outlier. Values greater than 13.5 are outliers.

* BloodPressure's values less than 40.0 and greater than 104.0 are outliers.

* SkinThickness's values less than 9.5 and greater than 45.5 are outliers.

* BMI has no low-outlier. Values greater than 50.25 are outliers.

* DiabetesPedigreeFunction has no low-outlier. Values greater than 1.201 are outliers.

* Age has no low-outlier. Value greater than 66.5 are outliers.

## Feature Scaling & Prepare Dataset

* Split data into Train and Test Set
  * Test size = 0.3, randomstate = 1612, stratify = y
* Apply StandardScaler

## Train and Evaluate Classification Model

### Support Vector Machine 

#### Hyper-parameters

```
C_values = [0.01, 0.1, 1] # from 0.01 to 1
gamma_values = [0.01, 0.1, 1]
kernel_values = ['linear', 'poly', 'rbf']

param_grid = {
    'kernel': kernel_values,
    'C': C_values,
    'gamma': gamma_values
}
```

#### GridSearch
```
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

model = SVC(random_state=1612)
grid = GridSearchCV(model, param_grid, cv=2)
grid.fit(X_train_scale, y_train)

svc = grid.best_estimator_
```

#### Metric scores
```
print('Accuracy score on Train Set:', str(svc.score(X_train_scale, y_train)))
print('Accuracy score on Test Set:', str(svc.score(X_test_scale, y_test)))
```

```
Accuracy score on Train Set: 0.776536312849162
Accuracy score on Test Set: 0.7792207792207793
```

#### Confusion Matrix
```
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()
y_pred = svc.predict(X_test_scale)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, 
            cmap='Blues', fmt='.1f')
```

<img src="https://user-images.githubusercontent.com/70437668/141252349-cac693b9-9b2e-4edf-bb90-115aad939d54.jpg" width=50% height=50%>

### Decision Tree

#### Hyper-parameters

```
params = {
    'criterion': ['entropy','gini'],
    'max_depth': [3,5,7],
    'min_samples_split': np.linspace(0.1, 1.0, 10), 
    'max_features':  ['auto', 'log2']
}
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=1612)
decision_tree.fit(X_train_scale, y_train)
```

```
print('Accuracy on Train Set: ', decision_tree.score(X_train_scale, y_train))
print('Accuracy on Test Set: ', decision_tree.score(X_test_scale, y_test))
```

```
Accuracy on Train Set:  1.0
Accuracy on Test Set:  0.7186147186147186
```

#### Confusion Matrix
```
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()
y_pred = decision_tree.predict(X_test_scale)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, 
            cmap='Blues', fmt='.1f')
```

<img src="https://user-images.githubusercontent.com/70437668/141252230-4cc59c63-5357-4f42-ae55-323c4049fccf.jpg" width=50% height=50%>

### Deep Neural Network

#### Compile the DNN model
```
loss = []
acc = [] 
model = Sequential()
model.add(Dense(64, activation='relu', name='hidden_layer_1', input_shape=X_train_scale.shape[1:]))
model.add(Dense(64, activation='relu', name='hidden_layer_2', input_shape=X_train_scale.shape[1:]))
model.add(Dense(64, activation='relu', name='hidden_layer_3', input_shape=X_train_scale.shape[1:]))
model.add(Dense(1, activation='sigmoid', name='output_layer'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

model.summary()
```

```
Model: "sequential_10"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
hidden_layer_1 (Dense)       (None, 64)                576       
_________________________________________________________________
hidden_layer_2 (Dense)       (None, 64)                4160      
_________________________________________________________________
hidden_layer_3 (Dense)       (None, 64)                4160      
_________________________________________________________________
output_layer (Dense)         (None, 1)                 65        
=================================================================
Total params: 8,961
Trainable params: 8,961
Non-trainable params: 0
_________________________________________________________________
```

#### Train model
```
def train_model(model, epochs):
  history = model.fit(X_train_scale, y_train, epochs=epochs, verbose=0)
  print(model.evaluate(X_train_scale, y_train))  
  loss.extend(history.history['loss'])
  acc.extend(history.history['accuracy'])
```

#### Draw charts of Accuracy and Loss versus Epochs
```
def draw_chart(loss, acc):
  epochs = range(len(loss))

  plt.figure(figsize=(20,10))
  plt.subplot(1,2,1)
  plt.plot(epochs, loss, c='red')
  plt.title('Loss versus Epochs')

  plt.subplot(1,2,2)
  plt.plot(epochs, acc)
  plt.title('Accuracy versus Epochs')
```

```
train_model(model, epochs=150)
draw_chart(loss, acc)
```

```
17/17 [==============================] - 0s 1ms/step - loss: 0.0013 - accuracy: 1.0000
[0.0012705079279839993, 1.0]
```

![Loss](https://user-images.githubusercontent.com/70437668/141252202-ef06e2d2-239b-4993-8424-f092953040d5.jpg)

#### Evaluate the model
```
model.evaluate(X_train_scale, y_train)
model.evaluate(X_test_scale, y_test)
```

```
17/17 [==============================] - 0s 1ms/step - loss: 0.0013 - accuracy: 1.0000
8/8 [==============================] - 0s 2ms/step - loss: 2.1307 - accuracy: 0.7056
[2.130713939666748, 0.7056276798248291]
```


#### Confusion Matrix
```
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

y_pred = model.predict(X_test_scale)
y_pred = np.where(y_pred >= 0.5, 1, 0) # sigmoid returns range 0-1 so use np.where to return 0 & 1
#y_pred = np.argmax(y_pred, axis=1) # np.argmax used for multi-class classification
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues')
```

<img src="https://user-images.githubusercontent.com/70437668/141252187-7b3679a7-dde1-482e-9355-fc1c56d4ee37.jpg" width=50% height=50%>

### Confusion Matrices for 3 Models: Support Vector Machine, DecisionTree, Deep Neural Network

```
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()
plt.figure(figsize=(25,8))

y_pred_svc = svc.predict(X_test_scale)
cm_svc = confusion_matrix(y_test, y_pred_svc)
plt.subplot(1,3,1)
sns.heatmap(cm_svc, annot=True, 
            cmap='Blues', fmt='.1f')
plt.title('Confusion Matrix for SVC')

y_pred_decisiontree = decision_tree.predict(X_test_scale)
cm_decisiontree = confusion_matrix(y_test, y_pred_decisiontree)
plt.subplot(1,3,2)
sns.heatmap(cm_decisiontree, annot=True, 
            cmap='Blues', fmt='.1f')
plt.title('Confusion Matrix for Decision Tree')

y_pred_dnn = model.predict(X_test_scale)
y_pred_dnn = np.where(y_pred_dnn >= 0.5, 1, 0) 
cm_dnn = confusion_matrix(y_test, y_pred_dnn)
plt.subplot(1,3,3)
sns.heatmap(cm_dnn, annot=True, fmt='.1f', cmap='Blues')
plt.title('Confusion Matrix for DNN')
```

![Matrices](https://user-images.githubusercontent.com/70437668/141252161-717d16a9-aaf8-4492-ad5c-84a4409e9003.jpg)

## Conclusion
**According to the Accuracies on the Train Set, SVC model gains the highest score among 3 models (SVC, DecisionTree, DNN) at 0.7792; 0.7186; 0.7056 respectively. So the confusion matrix of SVC model has a better evaluation then the other 2. True Positives (TP): SVC model correctly predicts that patients do have diabetes with 132 patients, the highest of all 3 models.**

