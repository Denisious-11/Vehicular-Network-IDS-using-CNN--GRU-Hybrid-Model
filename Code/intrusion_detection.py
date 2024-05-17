# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GRU, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

import joblib



# Readiing dataset
train_df = pd.read_csv('../Dataset/train_set.csv')
print('Train dataset\n',train_df.head())

# Shape of the dataset
print('Shape: ',train_df.shape)


# Checking null values
print('Null values: \n',train_df.isna().sum())

# Dataset head
train_df.head()

# Column names
print('Column names:',train_df.columns)

# Train dataset information
train_df.info()

# Total count of each values from the column 'type'
print('Total count of each values from the column ',train_df['type'].value_counts())

# Total count of values ifrom column 'label'
print('Total count of values ifrom column',train_df['label'].value_counts())

# Ploting the target variable
total = float(len(train_df['label']))
ax = sns.countplot(x='label', data=train_df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')

plt.show()

# Ploting the target variable
sns.countplot(x='type', data=train_df)
plt.xticks(rotation=80)
plt.show()

# Get object columns from dataset
objectColumns = train_df.select_dtypes(include=object).columns
print('Get object columns from dataset ',objectColumns)

# Get numericals columns from dataset
numericalColumns = train_df.select_dtypes(include=np.number).columns
print('Get numericals columns from dataset ',numericalColumns)

#Reading test dataset
test_df = pd.read_csv('../Dataset/test_set.csv')
print('Reading test dataset',test_df.head())

# Test dataset information
test_df.info()

# Shape of the test dataset
print('Shape of the test dataset ',test_df.shape)

# Columns of test dataset
test_df.columns

# Get common columns from train and tes dataset
common_columns = train_df.columns.intersection(test_df.columns)
print('Get common columns from train and tes dataset ',common_columns)

#Number of unique columns
common_columns.nunique()

# Remove uncommon column of train dataset
for remove in train_df.columns:
    if remove not in common_columns:
        train_df.drop(remove,axis=1,inplace=True)

# Dataset after remove unwanted columns
print('Dataset after remove unwanted columns ',train_df.head())

# Shape of the train dataset after removing columns
train_df.shape

# Get object columns of the train dataset after removing unwanted columns
objectColumns = train_df.select_dtypes(include=object).columns
objectColumns

# Taking numericals columns from dataset
numericalColumns = train_df.select_dtypes(include=np.number).columns
numericalColumns

# Converting categorical columns into numerical columns using label encoder()
le = LabelEncoder()
for obj in objectColumns:
    train_df[obj] = le.fit_transform(train_df[obj])

# Dataset after label encoding
print('Dataset after label encoding ',train_df.head())

# Total count of column type after label encoding
train_df['type'].value_counts()

# Unique values after label encoding
train_df['type'].unique()

# Converting type into two type as 0 & 1 here 5 to 0 and other values to 1
train_df['type'] = train_df['type'].map({5:0}).fillna(1)

# Unique values after converting
train_df['type'].unique()

# Converting float to integer
train_df['type'] = train_df['type'].astype(int)

# Value counts of column 'type'
train_df['type'] .value_counts()

# Correlation of the columns
train_df.corr()

# Correlation of columns with column 'type'
print('Correlation of columns with column  ',train_df.corr()['type'])

# Heat map of correlation data
plt.figure(figsize=(20,10))
sns.heatmap(train_df.corr(),annot=True,cmap='YlGnBu')
plt.show()

# Splitting training dataset for train test split
X = train_df.drop('type',axis=1)
y = train_df['type']

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Target value count
print('Target value count ',y_train.value_counts())

# Balancing the training data using SMOTE
sm = SMOTE()
X_train_sm,y_train_sm = sm.fit_resample(X_train,y_train)

# Target value counts after SMOTE
print('Target value counts after SMOTE ',y_train_sm.value_counts())

# Feature selection score
chi_score = chi2(X_train_sm,y_train_sm)
chi_score

# Select best features from train datas using SelectKBest()
k_best_selector = SelectKBest(score_func=chi2, k=10)

# Saving feature selector model
# joblib.dump(k_best_selector,'SavedFiles/features.pkl')

# Fitting selected features
X_train_selected = k_best_selector.fit_transform(X_train_sm, y_train_sm)

# Fitting selected features train data target values
X_test_selected = k_best_selector.transform(X_test)

# Indices f selected features
selected_feature_indices = k_best_selector.get_support(indices=True)
print("Selected feature indices: ", selected_feature_indices)

# Feature scaling the train data
scale = StandardScaler()
X_train_sc = scale.fit_transform(X_train_selected)
X_test_sc = scale.fit_transform(X_test_selected)

# Saving the feature scaling
# joblib.dump(scale,'/content/drive/MyDrive/Vehicle/scale.pkl')

# Shape after scaling
X_train_sc.shape

X_test_sc.shape

# Reshaping the train data for model training
X_train_reshape = np.reshape(X_train_sc, (X_train_sc.shape[0], X_train_sc.shape[1], 1))
print('Reshaping the train data for model training ',X_train_reshape.shape)

# Reshaping the test data for model training
X_test_reshape = X_test_sc.reshape(X_test_sc.shape[0],X_test_sc.shape[1],1)
print('Reshaping the test data for model training ',X_test_reshape.shape)

# CNN-GRU model architecture for training
model = Sequential([
    Conv1D(64,3,activation='relu',padding='same',kernel_regularizer=l2(0.03),input_shape=(X_train_reshape.shape[1],1)),
    BatchNormalization(),
    Dropout(0.2),
    GRU(128,activation='relu',return_sequences=True),
    Flatten(),
    Dense(1,activation='sigmoid')
])

# Architecture summary
model.summary()

# Model compiling
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=(['accuracy']))

# Saving accracy checkpoints
checkpoint_path = "SavedFiles/model.h5"  # Specify the path to save the checkpoints
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)

# Training the model
history = model.fit(X_train_reshape,y_train_sm,batch_size=120,epochs=50,validation_data=(X_test_reshape,y_test),callbacks=([checkpoint_callback]))

# Saving model
path = 'SavedFiles/arch.json'
arch = model.to_json()
with open(path, 'w') as json_file:
    json_file.write(arch)
