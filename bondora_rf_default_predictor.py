# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from time import time
from collections import defaultdict

class BondoraPredictor:
    loans_file = r'input_data\LoanData.csv'

    # define all data columns to use
    #used_cols = ['LoanApplicationStartedDate','DefaultDate','ApplicationSignedHour','ApplicationSignedWeekday','VerificationType',
    #'Age','Gender','LoanDuration','Country','Rating_V2', 'County', 'City',
    #'UseOfLoan', 'Education', 'MaritalStatus', 'EmploymentStatus','EmploymentDurationCurrentEmployer',
    #'EmploymentPosition','WorkExperience','OccupationArea','HomeOwnershipType','IncomeFromPrincipalEmployer',
    #'IncomeFromPension','IncomeFromFamilyAllowance','IncomeFromLeavePay','IncomeFromChildSupport','IncomeOther',
    #'IncomeTotal','ExistingLiabilities','LiabilitiesTotal','RefinanceLiabilities','DebtToIncome','FreeCash',
    #'MonthlyPaymentDay','ProbabilityOfDefault'
    #]
    used_cols = ['LoanApplicationStartedDate','DefaultDate','Rating_V2',
                 'ProbabilityOfDefault','FreeCash','DebtToIncome','LiabilitiesTotal','IncomeTotal']

    # define columns for encoding
    #cat_cols = ['Country','Rating_V2','County','City','EmploymentDurationCurrentEmployer','EmploymentPosition','WorkExperience']
    cat_cols = ['Rating_V2']

    # label column
    label_col = 'Status'
    
    #DataFrame column order
    cols = []

    def __init__(self):

        # import CSV data
        t0 = time()
        df = pd.read_csv(BondoraPredictor.loans_file,
                         usecols = BondoraPredictor.used_cols,
                         parse_dates=['LoanApplicationStartedDate','DefaultDate'],
                         infer_datetime_format=True)

        # filter only loans older than a year
        df = df[(df['LoanApplicationStartedDate'] < '2016-06-01')]
        #df.info()

        # define good/bad loans, good=true (defaultDate is not set)
        df['Status'] = df['DefaultDate'].isnull()

        # drop datafields and rows, where some data is missing
        df = df.drop('LoanApplicationStartedDate', axis=1)
        df = df.drop('DefaultDate', axis=1)
        df = df.dropna()

        # encode all used non-number column content
        self.labelEncoders = defaultdict(preprocessing.LabelEncoder)
        for c in BondoraPredictor.cat_cols:
            df[c] = self.labelEncoders[c].fit_transform(df[c])
            print(self.labelEncoders[c].classes_)
            
        
        # take sample for training
        df_train_features = df.sample(frac=0.8, replace=True)
        df_train_labels = df_train_features[BondoraPredictor.label_col]
        df_train_features.drop(BondoraPredictor.label_col, axis=1, inplace=True)
        print("Training samples: ", df_train_features.shape)

        # take sample for testing
        df_test_features = df.sample(frac=0.2, replace=True)
        df_test_labels = df_test_features[BondoraPredictor.label_col]
        df_test_features.drop(BondoraPredictor.label_col, axis=1, inplace=True)
        print("Testing samples: ", df_test_features.shape)
        t0 = time()

        # train RF classifier
        self.clf = RandomForestClassifier(n_estimators=100)
        self.clf.fit(df_train_features, df_train_labels)
      
        self.cols = df_train_features.columns.values
        print("Training time: ", round(time()-t0, 3), "s")

        t0 = time()
        score = self.clf.score(df_test_features,df_test_labels)
        print("=================================")
        print("Prediction score: ", score)
        print("=================================")
        print("Predicting time: ", round(time()-t0, 3), "s")

        p = self.clf.predict(df_test_features)
        cm = confusion_matrix(df_test_labels, p)
        print(cm)

    def predict(self, features):
        prediction_input = []
        
        # Here we guarantee the right order of feature values
        for c in self.cols:
            prediction_input.append(features[c]) 
                    
        result = self.clf.predict([prediction_input])[0]
        return result

    # this method should be used to transform labes for prediction input data
    def transformValue(self, featureName, value):
        transformed_value = self.labelEncoders[featureName].transform(value)
        return transformed_value
