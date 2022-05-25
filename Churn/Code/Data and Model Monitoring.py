import time
import datetime as dt
import dateutil
from pyspark.sql.functions import broadcast
from pyspark import SparkConf,SparkContext
from pyspark.sql.context import SparkSession
from pyspark.sql.context import HiveContext
from pyspark.sql import functions as F
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType,StringType,DoubleType,DateType
from pyspark.sql.session import SparkSession as SS
from pyspark.sql.functions import when,upper,col,lit,months_between
from pyspark.sql.functions import *
from pyspark.sql.functions import isnan, when, count, col
from scipy.stats import wasserstein_distance
from utils import *

sqlContext = get_sqlContext(spark_queue = 'default')

class Monitoring:
    
    def __init__:
        self.db = db
        self.user = user
        self.data_loc = data_loc
        self.base_snapshot_date = base_snapshot_date
        self.snapshot_date = snapshot_date
        self.sqlContext = sqlContext

    def _get_params(self):
        
        params = {
            'db':self.db,
            'user':self.user,
            'data_loc':self.data_loc,
            'snapshot_date':self.snapshot_date,
            'curr_snapshot':self.snapshot_date.replace('-','_')[:7]
            'base_snapshot_date':self.base_snapshot_date,
            'base_snapshot':self.base_snapshot_date.replace('-','_')[:7]
        }
    
    #generate statistics between two distributions  
    
    def drift(self, col, base, curr):
        
        _,p_ttest = stats.ttest_ind(base, curr, equal_var = False)
        
        base = (base - np.min(base))/(np.max(base) - np.min(base))
        base[np.isnan(base)] = 0
        curr = (curr - np.min(curr))/(np.max(curr) - np.min(curr))
        curr[np.isnan(curr)] = 0
        
        _,p_ks_samp = ks_2samp(base, curr )
        
        ct = np.min([np.shape(base)[0],np.shape(curr)[0]])
        jensenshannon_dist = distance.jensenshannon(base[:ct],curr[:ct])
        
        dist_em = wasserstein_distance(base,curr)
        std = np.min([np.std(base),np.std(curr)])
        
        ks_test_2 = pd.DataFrame(krushall_jul_oct_2,columns = ['features', 'kolmogorov_p_value'])
        js_div_2 = pd.DataFrame(jensenshannon_div_2,columns = ['features', 'js_divergence']).sort_values('js_divergence',ascending = False)
        wasserstein_dist_2 = pd.DataFrame(em_dist_2,columns = ['features', 'wasserstein_distance','std']).sort_values('wasserstein_distance',ascending = False)

        return p_ttest, p_ks_samp, jensenshannon_dist, dist_em, std
        
        
    #iterate over all features    
    
    def feature_drift(self, params, required_feats):
        
        t_test = []
        ks_test = []
        jensenshannon_div_2 = []
        em_test_2 = []
        stds = []
        
        for feat in required_feats:
            
            base_data_query = """select * from {db}.{user}_churn_data_{base_snapshot}""".format(**params)
            (return_code, base_data) = run_query(self.spark_df, base_data_query)

            if(return_code !=0):
                return return_code, None

            curr_data_query = """select * from {db}.{user}_churn_data_{curr_snapshot}""".format(**params)
            (return_code, curr_data) = run_query(self.spark_df, curr_data_query)

            if(return_code !=0):
                return return_code, None
            
            
            base = np.nan_to_num(np.array(base_data.select(feat).rdd.flatMap(lambda x: x).collect()).astype(float),nan = 0)
            curr = np.nan_to_num(np.array(curr_data.select(feat).rdd.flatMap(lambda x: x).collect()).astype(float),nan = 0)
            p_ttest, p_ks_samp, jensenshannon_dist, dist_em, std = self.drift(col, base, curr)
            
            t_test.append(p_ttest)
            ks_test.append(p_ks_samp)
            jensenshannon_div_2.append(jensenshannon_dist)
            em_test_2.append(dist_em)
            stds.append(std)
            
        df = pd.DataFrame()
        df['feat'] = feat
        df['p-value t_test'] = feat
        df['p-value ks_test'] = feat
        df['jensenshannon distance'] = feat
        df['wasserstein_distance'] = em_test_2
        df['std'] = stds
        
        return 0,df
       
    # drift statistics for predictions
    
    def prediction_drift(self, params):
        
        t_test = []
        ks_test = []
        jensenshannon_div_2 = []
        em_test_2 = []
        stds = []
        
        base_data_query = """select * from {db}.{user}_churn_{base_snapshot}_predictions""".format(**params)
        (return_code, base_data) = run_query(self.spark_df, base_data_query)
        
        if(return_code !=0):
            return return_code, None
        
        curr_data_query = """select * from {db}.{user}_churn_{curr_snapshot}_predictions""".format(**params)
        (return_code, curr_data) = run_query(self.spark_df, curr_data_query)
        
        if(return_code !=0):
            return return_code, None
        
        
        base = np.nan_to_num(np.array(base_data.select(feat).rdd.flatMap(lambda x: x).collect()).astype(float),nan = 0)
        curr = np.nan_to_num(np.array(curr_data.select(feat).rdd.flatMap(lambda x: x).collect()).astype(float),nan = 0)
        p_ttest, p_ks_samp, jensenshannon_dist, dist_em, std = self.drift(col, base, curr)

        t_test.append(p_ttest)
        ks_test.append(p_ks_samp)
        jensenshannon_div_2.append(jensenshannon_dist)
        em_test_2.append(dist_em)
        stds.append(std)
        
        f = pd.DataFrame()
        df['feat'] = feat
        df['p-value t_test'] = feat
        df['p-value ks_test'] = feat
        df['jensenshannon distance'] = feat
        df['wasserstein_distance'] = em_test_2
        df['std'] = stds
        
        return 0, df
    
    #generate model performance and business metrics for each decile/percentile
    
    def _get_KS_metrics(self, params, windows):
        print("KS_global_metrics")
        params['obj_col'] = "customerID",
        params['label_col'] = "Churn",
        params['score_col'] = "probabilities",
        params['windows'] = str(windows)

        query = """
        select
        percentile,
        min(score_col) as minimum_score,
        max(score_col) as maximum_score,
        sum(case when label_col > 0 then 1 else 0 end) as events,
        sum(case when label_col = 0 then 1 else 0 end) as non_events,
        sum(case when lebel_col = 1 then MonthlyCharges else 0 end) as Churn_Monthly_dollar_value,
        sum(MonthlyCharges) as Overall_Monthly_dollar_value,
        sum(case when gender = "Female" and label_col = 1 then 1 else 0 end) as Female_Churn,
        sum(case when gender = "Male" and label_col = 1 then 1 else 0 end) as Male_Churn,
        sum(case when SeniorCitizen = "Yes" and label_col = 1 then 1 else 0 end) as Senior_Citizen_Churn,
        avg(tenure) as avg_tenure
        from
        (
            select *,
            {obj_col} as obj_col, {label_col} as label_col, {score_col} as score_col,
            CEIL((PERCENT_RANK() OVER (order by {score_col} desc) ) * {windows} ) as percentile
            from {db}.{user}_churn_{snapshot}_predictions_with_labels
        ) as x
        group by percentile
        """.format(**params)

        (result, data) = run_query(query, self.obj_spark)
        data = data.toPandas()
        data.sort_values('percentile', inplace = True)

        sum_Events = data['events'].sum()
        sum_NonEvents = data['non_events'].sum()

        data['Events_Dist'] = data['events']/sum_Events
        data['NonEvents_Dist'] = data['non_events']/sum_NonEvents

        Events_Cumm = calculate_cumm_dist(list(data['Events_Dist']))
        NonEvents_Cumm = calculate_cumm_dist(list(data['NonEvents_Dist']))

        data['Events_Cumm'] = Events_Cumm
        data['NonEvents_Cumm'] = NonEvents_Cumm
        data['Events_Sum'] = calculate_cumm_dist(list(data['events']))
        data['NonEvents_Sum'] = calculate_cumm_dist(list(data['non_events']))

        data['Precision'] = data['Events_Sum']/ (data['Events_Sum'] + data['NonEvents_Sum'])

        data['KS'] = (data['Events_Cumm'] - data['NonEvents_Cumm'])*100
        return data
    
    #generate AUC scores
    
    def _get_AUC_metrics(self, params):
        
        spark = get_spark_session()
        
        data = spark.sql("select * from {db}.{user}_churn_{snapshot}_predictions_with_labels".format(**params))
        data = data.na.fill(0)
        
        auc_roc = []
        auc_pr = []
        
        evaluator = BinaryClassificationEvaluator(labelCol=params['label_col']) #label columns
        result = data
        result = result.withColumn('rawPrediction',F.col("probabilities")) 

        auroc = evaluator.evaluate(result, {evaluator.metricName: "areaUnderROC"})
        auprc = evaluator.evaluate(result, {evaluator.metricName: "areaUnderPR"})
        auc_roc.append(auroc)
        auc_pr.append(auprc)
        df = pd.DataFrame()
        df['AUC-ROC'] = auc_roc
        df['AUC-PR'] = auc_pr
        df['snapshot'] = params['snapshot']
        
        return df
     
    #1 if drifting, 0 if test passes
    def _compare_drift_stats(self, params):
        pass
    
    #1 if drifting, 0 if test passes
    def _compare_KS(self, params):
        pass
    
    #1 if drifting, 0 if test passes
    def _compare_AUC(self, params):
        pass
    
    def wrapper(self):
        pass

