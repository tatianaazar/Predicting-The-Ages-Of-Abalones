# adapted from: https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
# and: https://keras.io/guides/

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


data = pd.read_csv("abalone.head")  # importing the dataset as 'data' 

data.columns = [ "Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]  # adding name to the columns in 'data'


for row in data.iterrows():   # iterating over each row in 'data' and adding 1.5 to the rings value to get the 'Age' column
    data['Age'] = data['Rings'] + 1.5

data['Sex'].replace(['M', 'F', 'I'], [0, 1, 2], inplace=True) # one hit encoding the Sex feature where 0 is for Males, 1 is for Females, and 2 is for Infants

cols_to_standardize = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'] # standardizing the data 


for col in cols_to_standardize:  ## the columns that need to be standardized
    stand_col = f"{col}_Standardized"
    data[stand_col] = (data[col] - data[col].mean()) / data[col].std()

seed = 42  # setting a seed here for reproducibility; 42 is used as the random_state by convention
np.random.seed(seed)
tf.random.set_seed(seed)

X1 = data[['Sex', 'Length_Standardized', 'Diameter_Standardized', 'Height_Standardized', 'Whole_weight_Standardized', 'Shucked_weight_Standardized', 'Viscera_weight_Standardized', 'Shell_weight_Standardized']] # defining the X values
y1 = data['Age'] # defining target variable Age

kmeans = KMeans(n_clusters=3, random_state=seed) # applying k-means clustering 
clusters = kmeans.fit_predict(X1) 

cluster_dum = pd.get_dummies(clusters, prefix='Cluster', drop_first=False) # to one-hot encode the cluster feature
cluster_dum = cluster_dum.astype(int)

X1 = pd.concat([X1.reset_index(drop=True), cluster_dum.reset_index(drop=True)], axis=1) # to concatenate the cluster variables to the X-values defined above 

X1_train_val, X1_test, y1_train_val, y1_test = train_test_split(X1, y1, test_size=0.10, random_state=seed) # to split the dataset into training and test sets

X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train_val, y1_train_val, test_size=0.10, random_state=seed) # to split the training set into training and validation sets


model = Sequential([  # defining the neural network model 
    Input(shape=(X1_train.shape[1],)), 
    Dense(64, activation='relu', kernel_regularizer=l1(0.01)),  # first hidden layer
    Dense(32, activation='relu'),  # second hidden layer
    Dropout(0.2),  # layer for dropout
    Dense(1)  # output layer fot regression
])

model.compile(loss="mean_squared_error", optimizer='adam', metrics=["mae"]) # to compile the model

history = model.fit(X1_train, y1_train, validation_data=(X1_val, y1_val), epochs=100, batch_size=32, verbose=0) # to train the model

y1_pred = model.predict(X1_test).flatten() # to make predictions on the test set

MAE = mean_absolute_error(y1_test, y1_pred) # metrics to evaluate the model's performance on the test set and print them
MSE = mean_squared_error(y1_test, y1_pred)
R2 = r2_score(y1_test, y1_pred)
print(f"Mean Absolute Error: {MAE:.2f}")
print(f"Mean Squared Error: {MSE:.2f}")
print(f"R-squared Score: {R2:.2f}")

