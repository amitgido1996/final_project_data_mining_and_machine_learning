import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta

from madlan_data_prep import prepare_data

# Read the Excel file
file_path = "output_all_students_Train_v10.xlsx"
df = pd.read_excel(file_path)

# Call the prepare_data function
processed_data = prepare_data(df)

# Dropping the columns with more than 15% missing values
processed_data.drop(['total_floors', 'number_in_street', 'publishedDays'], axis=1, inplace=True)

processed_data.drop(['num_of_images', 'hasElevator', 'hasParking', 'hasBars', 'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly', 'floor'], axis=1, inplace=True)

processed_data.drop(['condition', 'entranceDate', 'furniture'], axis=1, inplace=True)

#Dropping columns that no need (e.g. description)
processed_data.drop(['description'], axis=1, inplace=True)


X = processed_data.drop("price", axis=1)
y = processed_data.price


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O']
num_cols


cat_cols = [col for col in X_train.columns if (X_train[col].dtypes=='O')]
cat_cols


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median', add_indicator=False)),
    ('scaling', StandardScaler())
])


categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])


column_transformer = ColumnTransformer([
     ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols)
    ], remainder='drop')


from sklearn import linear_model
from sklearn.svm import LinearSVC
pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', linear_model.ElasticNet())
])


pipe_preprocessing_model.fit(X_train, y_train)


y_pred = pipe_preprocessing_model.predict(X_test)


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

# Create an ElasticNet model
model = ElasticNet()

# Perform cross-validation
scores = cross_val_score(pipe_preprocessing_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

# Calculate root mean squared error (RMSE)
rmse_scores = np.sqrt(-scores)

# Print the RMSE scores for each fold
print("RMSE scores:", rmse_scores)

# Calculate and print the average RMSE score
print("Average RMSE:", rmse_scores.mean())

##In this model we used a cross-validation feature, in our case this feature split the data into 10 parts
# where the number of parts equals the number of iterations - in each iteration one part is used for validation while
#the rest of the parts are used for training, when in each iteration we get Rmse accuracy for this iteration.
#It can be seen that The 6rd iteration, the trampling accuracy is the lowest and from this it can be concluded that the performance of
#the model is the best in this iteration,compared to the 9th iteration where the accuracy is the highest in the execution of this iteration less close to the actual values. ​​than in the 3rd iteration.
#An advantage of using the cv feature, it provides us with a stronger performance evaluation,
#a more reliable estimate and this can also reduce the changes in the model and allow maximum utilization of data.

import pickle

# After training the model
trained_model = pipe_preprocessing_model

# Save the trained model as a PKL file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(trained_model, file)


    
