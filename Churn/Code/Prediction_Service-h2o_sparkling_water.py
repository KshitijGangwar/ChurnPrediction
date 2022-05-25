import pandas as pd
import re
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import os
import json
from functools import reduce
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid import H2OGridSearch
import pyspark.sql.functions as F 
from pyspark.sql.functions import udf, when
from pysparkling.ml import *
from math import log10,floor,ceil

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Python Spark SQL Hive integration example").enableHiveSupport().getOrCreate()

import sys
sys.path.insert(1,'/ads_storage/shared_libraries/odbc_new_cluster_coe/')
import HiveODBC as odbc
import json
from utils import *

class Predictions:
    
    def __init__:
        self.db = db
        self.user = user
        self.hdfs_loc = hdfs_loc
        self.snapshot_date = snapshot_date
        self.model_loc = model_loc
        self.model_name = model_name

    def _get_params(self):
        
        params = {
            'db':self.db,
            'user':self.user,
            'hdfs_loc':self.hdfs_loc,
            'snapshot_date':self.snapshot_date,
            'model_loc':self.model_loc,
            'model_name':self.model_name,
            'snapshot':self.snapshot_date.replace('-','_')[:7]
        }
        
        return params
    
    def preprocess_data(data):
        #preprocesing_here
        return data
    
    def get_data(self, fname):
        data = spark.sql("select * from {table_name}".format(**{'table_name':fname}))
        data = preprocess_data(data)
        return data

    def load_models(self, fname, model_name):
        path = fname+model_name
        settings = H2OMOJOSettings(predictionCol = 'pred_class',detailedPredictionCol = 'probs',withDetailedPredictionCol = True,convertUnknownCategoricalLevelsToNa = True, convertInvalidNumbersToNa = True)
        model = H2OMOJOModel.createFromMojo(path, settings)
        return model
    
    def get_prediction(self, model, data):
        data = model.transform(data)
        pred_class = 'probabilities'
        data = data.withColumn(pred_class,F.col("probs.probabilities.1"))
        return data

    def save_results(self, data, table_name, location_name):
        data.registerTempTable("dataTable")
        drop_query = "DROP TABLE  IF EXISTS " + table_name
        print(drop_query)
        spark.sql(drop_query)
        create_query = "CREATE TABLE " + table_name + " location '" + location_name + "' AS SELECT * from dataTable"
        print(create_query)
        spark.sql(create_query)
        return True

    def wrapper(self):
        try:
            params = self.get_params()
            feature_table = ["""{db}.{user}_churn_data_{snapshot}""".format(**parameters)]
            predictions_table = """{db}.{user}_churn_{snapshot}_predictions""".format(**parameters)

            model = self.load_models(params['model_path'], params['model_name']) 
            data = self.get_data(feature_table)
            results = self.get_prediction(model, data)
            self.save_results(results, predictions_table, params['model_loc']+'/'+result_table)
            return 0
        except:
            return 1      
        

