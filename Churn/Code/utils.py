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


def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


def roc_curve_show(y_test, y_pred_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--' )
    plt.plot(fpr, tpr, label=model_name,color = "r")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + 'ROC Curve',fontsize=16)
    plt.show()
    
def pr_curve_show(y_test, y_pred_prob, model_name):
    p, r, thresholds = precision_recall_curve(y_test, y_pred_prob)
    #plt.plot([0, 1], [0, 1], 'k--' )
    plt.plot(p, r, label=model_name,color = "r")
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title(model_name + 'PR Curve',fontsize=16)
    plt.show()
    
def calculate_cumm_dist(a):
    l = []
    for i in range(len(a)):
        if i == 0:
            l.append(a[i])
        else:
            x = sum(l)
            l.append(a[i]+l[-1])

    return l

def spark_context_file(SparkConf,Appname,type_conf="M",queue='user.H20.aig_scaled'):
    if type_conf=="S":
        conf= SparkConf().setAppName(Appname).set("spark.sql.shuffle.partitions",1000).        set("spark.executor.instances","15").set("spark.executor.cores","3").set('spark.yarn.queue',queue)
    elif type_conf=="M":
        conf = (SparkConf()
        .setAppName(Appname)
        .set("spark.broadcast.factory","org.apache.spark.broadcast.HttpBroadcastFactory")\
        .set("spark.yarn.queue",queue).set("spark.memory.useLegacyMode","true")\
        .set("spark.dynamicAllocation.enabled", "true")\
        .set("spark.shuffle.service.enabled","true").set("spark.sql.shuffle.partitions","1500").set("spark.storage.memoryFraction","0.1")\
        .set("spark.executor.cores", "4").set("spark.shuffle.memoryFraction","0.1").set("spark.executor.instances","25")\
        .set("spark.executor.memory","40g")\
        .set("spark.driver.memory","40g")\
        .set("spark.scheduler.minRegisteredResourcesRatio", "0.4").set("yarn.nodemanager.vmem-check-enabled","false")\
        .set("spark.network.timeout", "72000s").set("spark.yarn.executor.memoryOverhead","1g").set("spark.yarn.driver.memoryOverhead","512m")\
        .set("spark.sql.autoBroadcastJoinThreshold","10000000000").set("spark.dynamicAllocation.executorIdleTimeout","10800s")\
        .set("spark.shuffle.registration.timeout","2m").set("spark.shuffle.sasl.timeout","240s").set("spark.network.auth.rpcTimeout","240s")\
        .set("spark.rpc.askTimeout","7200s").set("spark.rpc.lookupTimeout","7200s")\
        .set("spark.task.maxFailures","8").set("spark.rpc.io.serverTreads","64").set("spark.executor.extraJavaOptions","-XX:ParallelGCThreads=4 -XX:+UseParallelGC")\
        .set("spark.shuffle.io.backLog","8192").set("spark.shuffle.io.serverThreads","128").set("spark.shuffle.registration.maxAttempst","5")\
        .set("spark.shuffle.io.maxRetries","100").set("spark.dynamicAllocation.maxExecutors", "200").set("setLogLevel","DEBUG")
        )

    else:
        raise ValueError("No Pyspark configuration")
    return conf


def get_sqlContext(spark_queue = 'user.H20.aig_scaled'):
    
    appname="Churn_Monitoring"
    SparkContext.setSystemProperty('spark.speculation','True')
    # conf = spark_context_file(SparkConf,appname,type_conf="S",queue=spark_queue)
    conf = spark_context_file(SparkConf,appname,type_conf="M",queue=spark_queue)
    #spark = SparkSession.builder.config('spark.port.maxRetries', 100).getOrCreate()
    spark = SS.builder.config(conf=conf).getOrCreate()
    
    sc = spark.sparkContext
    #print(sc._conf.getAll())
    
    sqlContext = SQLContext(sc)
    
    return sqlContext

def get_spark_session():
    spark_config = [("spark.driver.memory","5g"),                ("spark.executor.memory","15g"),                ("spark.executor.cores","5"),                ("spark.yarn.queue","root.user.H20.default")] 
    spark = SparkSession.builder.appName("Python Spark SQL Hive integration example 1").master("yarn")        .config(conf=SparkConf().setAll(spark_config)).enableHiveSupport().getOrCreate()
    
    return spark

def run_query(sqlContext, query):
    
    for q in query.split(';'):
        try:
            spark_df = sqlContext.sql(q)
        except:
            return (1, None)

        
    return (0, spark_df)