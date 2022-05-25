import h2o
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from datetime import date

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid import H2OGridSearch
from h2o.automl import H2OAutoML

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import math
import os

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

from pandas import ExcelWriter

from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix, precision_recall_curve,auc
from sklearn.metrics import r2_score

from utils import *

h2o.init(strict_version_check=False)

class Training:
    
    def __init__:
        self.version_name = version_name
        self.model_train_directory = model_train_directory
        self.train_data_loc = train_data_loc
        self.test_data_loc = train_data_loc
        self.model_loc = model_loc
        self.snapshot_date = snapshot_date
        self.train_predictions_loc = train_predictions_loc
        self.test_predictions_loc = test_predictions_loc
        self.train_KS_loc = train_KS_loc
        self.test_KS_loc = test_KS_loc
        self.train_AUC_loc = train_AUC_loc
        self.test_AUC_loc = test_AUC_loc
        self.feat_imp_loc = feat_imp_loc
        

    def _get_params(self):
        
        params = {
            'version_name' : self.version_name,
            'model_train_directory' : self.model_train_directory,
            'train_data_loc':self.train_data_loc,
            'test_data_loc':self.test_data_loc,
            'model_loc':self.model_loc,
            'snapshot_date':self.snapshot_date,
            'train_predictions_loc':self.train_predictions_loc,
            'test_predictions_loc':self.test_predictions_loc,
            'train_KS_loc':self.train_KS_loc,
            'test_KS_loc':self.test_KS_loc,
            'train_AUC_loc':self.train_AUC_loc,
            'test_AUC_loc':self.test_AUC_loc
            'feat_imp_loc':self.feat_imp_loc
        }
        
    def _create_version_dir(self, params):
        try:
            subprocess.popen("""mkdir -p {model_train_directory}/{version_name}""".format(**params))
            return 0
        except:
            return 1
        
    def manipulate(self, data):
        #data = data.drop(['customerID'], axis = 1)
        data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce')
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data.dropna(inplace = True)
        return data
    
    def preprocess_data(self, data):
        data = data.apply(lambda x: object_to_int(x) if x.name != 'time' else x)

        #feature_cols
        feature_cols = categorical_cols + numerical_cols

        scaler= StandardScaler()

        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        return data
    
    def get_feature_cols(self):
        numerical_cols = ['tenure', 'MonthlyCharges']
        categorical_cols = 'Contract', 'PaperlessBilling', 'OnlineSecurity', 'TechSupport']
        feature_cols = numerical_cols + categorical_cols
        return numerical_cols, categorical_cols, feature_cols
        
    def _get_data(self, params, split = 'train'):
        if(split=='train'):
            data = pd.read_csv(params['train_data_loc'])
        else(split=='test'):
            data = pd.read_csv(params['test_data_loc'])
        
        data = self.manipulate(data)
        data = self.preprocess_data(data)
        
        return data
    
    def get_automl_class(self, train_data, valid_data, feature_cols,actualCol, _max_runtime_secs, _max_runtime_secs_per_model):
        aml = H2OAutoML(project_name=actualCol, balance_classes = True, max_models = 5,
                    include_algos=['XGBoost','GBM','GLM','DRF', 'StackedEnsemble'], nfolds = 5, #,'DRF','GLM','StackedEnsemble'
                    max_runtime_secs=_max_runtime_secs, max_runtime_secs_per_model=_max_runtime_secs_per_model,
                    sort_metric='aucpr')
        aml.train(x=feature_cols,y=actualCol, training_frame=train_data, leaderboard_frame=valid_data, validation_frame=valid_data)
        lb = aml.leaderboard
        print ('Validation Scores')
        print(lb.head())
        best_model = aml.leader
        all_models = [h2o.get_model(id) for id in aml.leaderboard.as_data_frame().model_id]
        return best_model, all_models
    
    def train_models(self, label_col = 'Churn', train_val_split = 0.2, _max_runtime_secs = 2000, _max_runtime_secs_per_model = 400):
        
        data = self.get_data(self, params, 'train')
        feature_cols = self.get_feature_cols()
        
        Train_Network_File = data.sample(int(len(data)*train_val_split))
        Val_Network_File = data[~data['customerID'].isin(Train_Network_File['customerID'].unique())]

        del(data)
        cols = feature_cols
        cols.append(label_col)

        print('loading_train_frame...')
        train_data = h2o.H2OFrame(Train_Network_File[cols])
        print('loading_test_frame...')
        val_data = h2o.H2OFrame(Val_Network_File[cols])
        
        del(Train_Network_File)
        del(Val_Network_File)

        train_data[label_col] = train_data[label_col].asfactor()
        val_data[label_col] = val_data[label_col].asfactor()

        print('training_model')
        best_model, all_models = self.get_automl_class(train_data, val_data, feature_cols, label_col, _max_runtime_secs, _max_runtime_secs_per_model)

        return train_data, val_data, best_model, all_models

    def save_best_model(self, best_model, params):
        best_model.download_mojo(params['model_loc'])
        return 0


    def get_feature_importance(self, best_model):
        print("get_feature_importance")
        feature_imp = best_model.varimp(True)
        return feature_imp


    def save_feature_importance(self, feature_imp, params):
        print("save_feature_importance")
        feature_imp.to_csv(params['feature_imp_loc'] , index = False)
        return 0                                     


    def score(self, best_model, label_col = 'Churn', split = 'train'):
        print("scoring")
        feature_cols = self.get_feature_cols()
        data = get_data(self, params)
        
        df = h2o.H2OFrame(data[feature_cols])
        pred = best_model.predict(df)
        df["prediction"] = pred['p1']
        dt = df.as_data_frame()                 
        dt[label_col] = data[label_col]
        
        return dt
    
    def save_predictions(self, df, split = 'train'):
        if(split=='train'):
            data.to_csv(params['train_predictions_loc'], index = False)
        else(split=='test'):
            data.to_csv(params['test_predictions_loc'], index = False)
            
    def ks(self, data_df, label_col='Churn'):
        p = pd.DataFrame(data_df['prediction'])
        p.columns = ['pred']
        p['actuals'] = list(data_df[label_col])
        binned_x = pd.qcut(p['pred'], 10, duplicates = 'drop')
        p['binned'] = binned_x

        p_ = pd.DataFrame(p.groupby('binned',as_index='False').agg({'actuals':['sum','count']}))
        p_['binned'] = p_.index
        p_ = p_.reset_index(drop = True)
        p_.columns = ['Events','count','binned']


        p_['lower_limit'] = p_['binned'].apply(lambda x:float(str(x)[1:-1].split(', ')[0]))
        p_['upper_limit'] = p_['binned'].apply(lambda x:float(str(x)[1:-1].split(', ')[1]))
        p_ = p_.sort_values(by = 'upper_limit', ascending = False)
        p_['NonEvents'] = p_['count'] - p_['Events']
        sum_Events = p_['Events'].sum()
        sum_NonEvents = p_['NonEvents'].sum()
        p_['Events_Dist'] = p_['Events']/sum_Events
        p_['NonEvents_Dist'] = p_['NonEvents']/sum_NonEvents
        Events_Cumm = calculate_cumm_dist(list(p_['Events_Dist']))
        NonEvents_Cumm = calculate_cumm_dist(list(p_['NonEvents_Dist']))

        Events_Cumm_num = calculate_cumm_dist(list(p_['Events']))
        NonEvents_Cumm_num = calculate_cumm_dist(list(p_['NonEvents']))

        p_['Events_Cumm'] = Events_Cumm
        p_['NonEvents_Cumm'] = NonEvents_Cumm

        p_['Events_Cumm_num'] = Events_Cumm_num
        p_['NonEvents_Cumm_num'] = NonEvents_Cumm_num

        p_['KS'] = (p_['Events_Cumm'] - p_['NonEvents_Cumm'])*100
        p_['Precision'] = p_['Events_Cumm_num'] / (p_['Events_Cumm_num'] + p_['NonEvents_Cumm_num'])
        p_['Capture'] = p_['Events_Cumm']
        return(p_)

    def save_KS(self, df, params, split = 'train'):
        if(split == 'train'):
            with ExcelWriter(params['train_KS_loc']) as writer:
                df.to_excel(writer,split)
                writer.save()
        else if(split == 'test'):
            with ExcelWriter(params['test_KS_loc']) as writer:
                df.to_excel(writer,split)
                writer.save()
        return 

    def get_AUC_Metrics(self, df, label_col = 'Churn'):
        print("AUC-ROC: "+str(roc_auc_score(df[label_col], df['prediction'])))
        precision, recall, thresholds = precision_recall_curve(df[label_col], df['prediction'])
        area = auc(recall, precision)
        print("AUC-PR: "+str(area))
        return  
    
    def wrapper(self):
        pass
    
    




