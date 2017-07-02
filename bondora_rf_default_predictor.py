# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from time import time
from collections import defaultdict
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np

class BondoraPredictor:
    loans_file = r'input_data\LoanData.csv'

    # define all data columns to use https://api.bondora.com/doc/ResourceModel?modelName=Auction&v=1
    # excluded: DateOfBirth & ListedOnUTC. When using this these should be converted to int/float (days from epoch)
    used_cols = ['UserName','NewCreditCustomer','ApplicationSignedHour',
    'ApplicationSignedWeekday','VerificationType','LanguageCode','Age','Gender','Country',
    'AppliedAmount','Interest','LoanDuration','County','City','Education','EmploymentDurationCurrentEmployer',
    'HomeOwnershipType','IncomeTotal','MonthlyPaymentDay','ModelVersion','ExpectedLoss',
    'Rating_V2','LossGivenDefault','ProbabilityOfDefault','LiabilitiesTotal']

    # Currently country based credit ratings are excluded. Should test country based models
    # 'CreditScoreEsMicroL','CreditScoreEsEquifaxRisk','CreditScoreFiAsiakasTietoRiskGrade','CreditScoreEeMini',
    
    # MonthlyPayment - this is just often missing

    # Minimal set of features used previously
    # used_cols = ['LoanApplicationStartedDate','DefaultDate','Rating_V2',
    #             'ProbabilityOfDefault','FreeCash','DebtToIncome','LiabilitiesTotal','IncomeTotal']
    
    # values used for input_data, but not in predicting
    used_cols.append('LoanApplicationStartedDate')
    used_cols.append('DefaultDate')


    # label column
    label_col = 'Status'

    #DataFrame column order
    cols = []

    def __init__(self):

        # import CSV data
        t0 = time()
        df = pd.read_csv(BondoraPredictor.loans_file,
                         usecols = BondoraPredictor.used_cols,
                         low_memory=False,
                         parse_dates=['LoanApplicationStartedDate','DefaultDate'],
                         converters={'County': str, 'City': str, 'EmploymentDurationCurrentEmployer': str, 
                         'Rating_V2' : str},
                         #, 'CreditScoreEsMicroL': str,'CreditScoreEsEquifaxRisk': str,'CreditScoreFiAsiakasTietoRiskGrade': str},
                         compression='infer',
                         infer_datetime_format=False)

        # Convert listedOnUTC to days of the year

        # filter only loans older than a year
        df = df[(df['LoanApplicationStartedDate'] < '2016-06-01')]
        df['DebtToIncome'] = df['LiabilitiesTotal']/df['IncomeTotal']
        #for i in range(1,11):
        #    e = 0.1*i
        #    s = e-0.1
        #    print("binning {}-{} [{}]".format(s,e,i))
        #    df.loc[(df['ExpectedLoss'] > s) & (df['ExpectedLoss'] <= e), 'ExpectedLoss'] = i

        # define good/bad loans, good=true (defaultDate is not set)
        df['Status'] = df['DefaultDate'].isnull()

        # drop datafields and rows, where some data is missing
        df = df.drop('LoanApplicationStartedDate', axis=1)
        df = df.drop('DefaultDate', axis=1)
        df = df.dropna()


        # encode all used non-number column content
        self.labelEncoders = defaultdict(preprocessing.LabelEncoder)

        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = self.labelEncoders[c].fit_transform(df[c])

        df.info()
        gp = df.loc[df['Status'] == True, 'Status'].count()/df.shape[0]
        print("General probability of good loan: %0.2f" % gp)

        
        for rating in self.labelEncoders['Rating_V2'].classes_:
            r2_idx = self.transformValue('Rating_V2',[rating])[0]
            gp = df[(df['Status'] == True) & (df['Rating_V2'] == r2_idx)].shape[0] / df[df['Rating_V2'] == r2_idx].shape[0]
            print("... of good %s loan: %0.2f" % (rating, gp))

        ratings_for_preditiction = self.transformValue('Rating_V2',['A','AA','B'])

        gp = df[(df['Status'] == True) & (df['Rating_V2'].isin(ratings_for_preditiction))].shape[0] / df[df['Rating_V2'].isin(ratings_for_preditiction)].shape[0]
        print("A,AA,B general probability of good loan: %0.2f" % gp)

        # train RF classifier
        df = df[df['Rating_V2'].isin(ratings_for_preditiction)]
        X = df.drop(self.label_col, axis=1)
        y = df[self.label_col]
        cols = list(df.columns.values)
        self.clf = RandomForestClassifier(n_estimators=100)
        precision = cross_val_score(self.clf, X, y, cv=5, scoring='precision')
        #precision = tp/tp+fp
        print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))

        recall = cross_val_score(self.clf, X, y, cv=5, scoring='recall')
        #recall = tp/tp+tn
        print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))

        self.clf.fit(X, y)
        print("Training time: ", round(time()-t0, 3), "s")

        # Print the feature ranking
        importances = self.clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(X.shape[1]):
            print("%d. %s (%f)" % (f + 1, df.columns.values[indices[f]], importances[indices[f]]))


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


bp = BondoraPredictor()