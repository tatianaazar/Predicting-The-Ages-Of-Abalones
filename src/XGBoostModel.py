# Code adapted from: https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
# and https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


data = pd.read_csv("abalone.head")  # importing the dataset as 'data' 


data.columns = [ "Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]  # adding name to the columns in 'data'

for row in data.iterrows():  # iterating over each row in 'data' and adding 1.5 to the rings value to get the 'Age' column
    data['Age'] = data['Rings'] + 1.5

data['Sex'].replace(['M', 'F', 'I'], [0, 1, 2], inplace=True)  # one hit encoding the Sex feature where 0 is for Males, 1 is for Females, and 2 is for Infants


cols_to_standardize = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']  # the columns that need to be stndardized

for col in cols_to_standardize:  # for loop to standardize the columns and each will now be X_Standardized in the data
    stand_col = f"{col}_Standardized"
    data[stand_col] = (data[col] - data[col].mean()) / data[col].std()


X = data[['Sex', 'Length_Standardized', 'Diameter_Standardized', 'Height_Standardized', 'Whole_weight_Standardized', 'Shucked_weight_Standardized', 'Viscera_weight_Standardized', 'Shell_weight_Standardized']] # defining the X values
y = data['Age']  # defining the target variable 


preprocessor = ColumnTransformer(  # to preprocess the data and transform it 
    transformers=[
        ('num', X.columns) 
    ]
)

model = XGBRegressor(objective='reg:squarederror', learning_rate=0.07, max_depth=4, n_estimators=175,) # defining the model with the parameters used 

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)]) # creating a pipeline to combine the preprocessor and the model 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) # splitting the data into training and testing sets, 0.9 and 0.1 respectively

pipeline.fit(X_train, y_train) # fitting the pipeline to the training data

y_pred = pipeline.predict(X_test) # making predictions on the test data

MAE = mean_absolute_error(y_test, y_pred) # to get the metrics: R^2, MAE, RMSE and print them
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
print(f"R-squared: {R2:.3f}")
print(f"Mean Absolute Error (MAE): {MAE:.3f}")
print(f"Mean Squared Error (MSE): {MSE:.3f}")


