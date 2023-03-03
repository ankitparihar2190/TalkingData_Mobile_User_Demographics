import pandas as pd 
import numpy as np
import sklearn
import xgboost
import mlxtend
from joblib import dump, load
from mlxtend.classifier import StackingCVClassifier
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from scipy.sparse import  hstack
from sklearn import model_selection
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import metrics
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    accuracy_score
)
from flask import Flask, request, jsonify, render_template
import pickle
import csv
import math 

app = Flask(__name__)
#loading age and gender prediction model
age_prediction_model = pickle.load(open('finalized_model_age.pkl', 'rb'))
gender_prediction_model = pickle.load(open('finalized_model_gender.pkl', 'rb'))


# Gender Prediction Module #

#reading preprocessed Data

test_data_gender_samples = pd.read_csv('test_gender_samples.csv', header = "infer")
finalGenderDf_with_prediction = pd.read_csv('originalset_gender_50Samples.csv', header = "infer")    

X_test = test_data_gender_samples.drop("gender", axis = 1)
X_test_csr = csr_matrix(X_test)

# Gender prediction
gender_pred = gender_prediction_model.predict(X_test_csr)

#output Generation logic
finalGenderDf_with_prediction['gender_pred'] = gender_pred
finalGenderDf_with_prediction['gender_pred'] = finalGenderDf_with_prediction['gender_pred'].map({1: 'male', 0:'female'})
finalGenderDf_with_prediction.to_csv("genderPrediction.csv", index = None)
# End of gender prediction Logic

# Age Prediction Logic block
#reading Pre-processed csv
age_test_samples = pd.read_csv('age_test_samples.csv', header = "infer")
finalageDf_with_prediction = pd.read_csv('originalset_Age_Samples.csv', header = "infer")

X_test_age = age_test_samples.drop(["train_test_flag","age"], axis = 1)
X_test_age_csr = csr_matrix(X_test_age)

age_pred = age_prediction_model.predict(X_test_age_csr)

#output presentation Logic
finalageDf_with_prediction['age_pred'] = age_pred
finalageDf_with_prediction['age_pred'] = finalageDf_with_prediction['age_pred'].apply(lambda x : math.ceil(x))
finalageDf_with_prediction.to_csv("ageprediction.csv",index=None)
 
# End of Age prediction Logic block


@app.route('/')
def home():
    
    return render_template('index.html')

    
@app.route('/predict', methods=['POST'])
def predict():
    gender = pd.read_csv('genderPrediction.csv') #reading gender prediction
    age = pd.read_csv('ageprediction.csv')# reading ageprediction
    return render_template('index.html', tables=[gender.to_html(),age.to_html()], titles=[''])


if __name__ == "__main__":

    app.run(debug=True,host="0.0.0.0")
