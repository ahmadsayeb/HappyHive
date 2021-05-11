import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV,LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
import random
from flask import Flask, request, jsonify
from flask import Flask
import re
from sklearn.svm import SVC
import time

app = Flask(__name__)

def change_label_to_educ(row):

    '''changing encoded values back to labels'''

    if row == 0:
        row = 'computer science'
    elif row == 1:
        row = 'engineering'
    elif row == 2:
        row = 'other'
    else:
        row='science'
        
    return row

#creating logistic regression model
def logistic_regression(df):
    
    df = df.sample(frac=1)
    #getting the x and y from the data
    x_df = df.drop(columns=['manager','senior'],axis=1)
    y_df = df[['manager']]
    #dividing into test and train
    X_train, X_test = x_df.iloc[:500], x_df.iloc[500:]
    y_train, y_test = y_df.iloc[:500], y_df.iloc[500:]
    
    #removing gender
    gender_test = x_df[['gender_F','gender_M']].iloc[500:].reset_index().drop('index',axis=1)
    education_test = x_df['education_label'].iloc[500:].reset_index().drop('index',axis=1)
    #Normalizing the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    
    #Creating the cross validation and training
    clf = LogisticRegressionCV(cv=5, random_state=0).fit(X_train_transformed,y_train)
    
    #predicting
    y_predicted = clf.predict(X_test_transformed)
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    
    #predict the probability
    y_probability = clf.predict_proba(X_test_transformed)
    
    print("The accuracy is: ", accuracy)
    
    
    #making dataframe of everything
    pre_df = pd.DataFrame(y_predicted,columns=['Manager'])
    prob_df = pd.DataFrame(y_probability,columns=['False','True'])
    final_result = pd.concat([gender_test,education_test,prob_df,pre_df],axis=1)
    
    return clf,final_result


#creating support vector machine model
def svm_model(df):

    #shuffling the data
    df = df.sample(frac=1)

    #getting the x and y from the data
    x_df = df.drop(columns=['manager','senior'],axis=1)
    y_df = df[['manager']]

    #dividing into test and train
    X_train, X_test = x_df.iloc[:500], x_df.iloc[500:]
    y_train, y_test = y_df.iloc[:500], y_df.iloc[500:]
    
    #removing gender
    gender_test = x_df[['gender_F','gender_M']].iloc[500:].reset_index().drop('index',axis=1)
    education_test = x_df['education_label'].iloc[500:].reset_index().drop('index',axis=1)

    #Normalizing the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    
    #Creating the cross validation and training
    clf = SVC(kernel='poly',degree=3,probability=True).fit(X_train_transformed,y_train)
    
    #predicting
    y_predicted = clf.predict(X_test_transformed)
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    
    #predict the probability
    y_probability = clf.predict_proba(X_test_transformed)
    
    print("The accuracy is: ", accuracy)
    
    
    #making dataframe 
    pre_df = pd.DataFrame(y_predicted,columns=['Manager'])
    prob_df = pd.DataFrame(y_probability,columns=['False','True'])
    final_result = pd.concat([gender_test,education_test,prob_df,pre_df],axis=1)
    
    return clf,final_result

def linear_reg(df):
    
    df = df.sample(frac=1)
    #getting the x and y from the data
    x_df = df.drop(columns=['no_job_positions','senior'],axis=1)
    y_df = df[['no_job_positions']]
    #dividing into test and train
    X_train, X_test = x_df.iloc[:500], x_df.iloc[500:]
    y_train, y_test = y_df.iloc[:500], y_df.iloc[500:]
    
    #removing gender
    gender_test = x_df[['gender_F','gender_M']].iloc[500:].reset_index().drop('index',axis=1)
    education_test = x_df['education_label'].iloc[500:].reset_index().drop('index',axis=1)
    #Normalizing the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    
    #Creating the cross validation and training
    clf = LinearRegression().fit(X_train_transformed,y_train)
    
    #predicting
    y_predicted = clf.predict(X_test_transformed)
#     accuracy = metrics.accuracy_score(y_test, y_predicted)
    
    #predict the probability
    #y_probability = clf.predict_proba(X_test_transformed)
    
#     print("The accuracy is: ", accuracy)
    
    
    #making dataframe of everything
#     pre_df = pd.DataFrame(y_predicted,columns=['Manager'])
#     prob_df = pd.DataFrame(y_probability,columns=['False','True'])
#     final_result = pd.concat([gender_test,education_test,prob_df,pre_df],axis=1)
    
    return clf

def predict(clf, age, duration, no_company, no_job, gender_F, gender_M, education):

    '''predicting for the custom data'''

    # if education == 'science':
    #     education_label = 3
    # elif education == 'computer science':
    #     education_label = 0
    # elif education == 'other':
    #     education_label = 2
    # elif education == 'engineering':
    #     education_label = 1
    
    x = [age,duration, no_company, no_job, gender_F, gender_M,education]
    x = np.array(x)
    y = clf.predict(x.reshape(1,-1))
    
    return y


def predict_linear(model, age,duration,no_company, manager,gender_f, gender_M,education):
    x=[age,duration,no_company,manager, gender_f, gender_M, education]
    x = np.array(x)
    y = model.predict(x.reshape(1,-1))
    
    return y

def main():

    df = pd.read_csv('data/manual_encoded_multiplied.csv').drop('Unnamed: 0',axis=1)

    #removing space from computer science
    df['education'] = df['education'].apply(lambda x: 'computer science'\
                                             if x=='computer science ' else x)

    df['education_label'] = df['education_label'].apply(lambda x: 1 if x==0 else x)
    df['education_label'] = df['education_label'].apply(lambda x: x-1)

    df.drop(['education','linkedin link'],axis=1,inplace=True)
    df.dropna(how='any',inplace=True)

    #trying both models
    model,final_result = logistic_regression(df)
    model_svm, final_result_svm = svm_model(df)
    model_linear = linear_reg(df)

    #creating data frame out of test data and predicted values
    final_result_non_labeled = final_result.copy(deep=True)
    final_result_non_labeled['education_label'] = final_result['education_label']\
                                                .apply(change_label_to_educ)

    #creating data frame out test data and predicted values for svm
    final_result_non_labeled_svm = final_result_svm.copy(deep=True)
    final_result_non_labeled_svm['education_label'] = final_result_svm['education_label']\
                                                    .apply(change_label_to_educ)


    #predicting for custom values using both logistic and svm
    y_log = predict(model, 30, 1.5, 5, 2, 1,0, 1)
    y_svm = predict(model_svm, 35, 1.5, 5, 2, 1,0, 1)
    
    print('prediction_log:', y_log)
    print('prediction_svm:', y_svm)

    @app.route('/manager',methods=['GET'])
    def get_api_recommendation():
        x = []
        age = int(request.args.get('age'))
        duration = int(request.args.get('duration'))
        no_company = int(request.args.get('no_company'))
        no_job = int(request.args.get('no_job'))
        gender_f = int(request.args.get('gender_f'))
        gender_m = int(request.args.get('gender_m'))
        education = int(request.args.get('education'))
        x = [age,duration,no_company,no_job,gender_f,gender_m,education]
        y = predict(model,age,duration,no_company,no_job,gender_f,gender_m,education)
        return {'prediction': str(y[0])}

    @app.route('/movement')
    def get_api_movement():
        x = []
        age = int(request.args.get('age'))
        duration = int(request.args.get('duration'))
        no_company = int(request.args.get('no_company'))
        manager = int(request.args.get('manager'))
        gender_f = int(request.args.get('gender_f'))
        gender_m = int(request.args.get('gender_m'))
        education = int(request.args.get('education'))
        #x = [age,duration,no_company,no_job,gender_f,gender_m,education]
        y = predict_linear(model_linear,age,duration,no_company,manager,gender_f,gender_m,education)
        y = np.rint(y)
        return {'prediction': str(y[0])}


if __name__=="__main__":
    main()

    time.sleep(3)
    
    app.run()


    