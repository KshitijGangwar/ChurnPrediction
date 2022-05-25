import pandas as pd
import numpy as np
from utils import *

class ChurnDataFeatureStore:
    
    def __init__(self, db, user, data_loc, snapshot_date, obj_spark):
        self.db = db
        self.user = user
        self.data_loc = data_loc
        self.snapshot_date = snapshot_date
        self.obj_spark = obj_spark

    def _get_params(self):
        
        params = {
            'db':self.db,
            'user':self.user,
            'data_loc':self.data_loc,
            'snapshot_date':self.snapshot_date,
            'date_2y_ago':, (pd.to_datetime(snapshot_date) - relativedelta(months=24) + relativedelta(day=1)).strftime('%Y-%m-%d')
            'date_1y_ago':, (pd.to_datetime(snapshot_date) - relativedelta(months=12) + relativedelta(day=1)).strftime('%Y-%m-%d')
            'snapshot':self.snapshot_date.replace('-','_')[:7]
            
        }
        
        return params

    # generate data for customers who are eligible to churn
    def _get_data(self, params):

        query = """ 
        drop table if exists {db}.{user}_churn_data_{snapshot};
        create table {db}.{user}_churn_data_{snapshot}
        stored as parquet location "{data_loc}/{user}_churn_data_{snapshot}"
        
        as
        select customerID, gender, SeniorCitizen, Partner, Dependents, tenure, 
        PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
        from
        
        (select c.customerID, c.gender, c.SeniorCitizen, c.Partner, c.Dependents, c.tenure, 
        p.PhoneService, p.MultipleLines, p.InternetService, p.OnlineSecurity, p.OnlineBackup, p.DeviceProtection,
        p.TechSupport, p.StreamingTV, p.StreamingMovies,
        t.Contract, t.PaperlessBilling, t.PaymentMethod, t.MonthlyCharges, t.next_txn_due_date, s.TotalCharges,
        months_between(t.max_txn_date, c.join_date) as tenure
        
        from
        (select * from core.customers) as c
        
        inner join [SHUFFLE]
        
        (select straight_join a.*, b.max_txn_date
        case when a.contract = "Month-to-month" then Transaction_Amount
        when a.contract = "One year" then Transaction_Amount/12
        when a.contract = "Two year" then Transaction_Amount/24
        else null end as MonthlyCharges,
        select a.customerID, 
        case when a.contract = "Month-to-month" then ADD_MONTHS(b.max_txn_date, 1)
        when a.contract = "One year" then ADD_MONTHS(b.max_txn_date, 12)
        when a.contract = "Two year" then ADD_MONTHS(b.max_txn_date, 24)
        else null end as next_txn_due_date
        from
        (select * from core.transactions where transaction_date between {date_2y_ago} and {snapshot_date}) as a
        left join [SHUFFLE]
        (select customerID, max(transaction_date) as max_txn_date from core.transactions where transaction_date between {date_2y_ago} and {snapshot_date}) as b
        
        on 
        a.customerID = b.customerID
        and a.transaction_date = b.max_txn_date
        ) as t
        
        on 
        c.customerID = t.customerID
        
        left join [SHUFFLE]
        
        (select customerID, sum(Transaction_Amount) as TotalCharges from core.transactions where transaction_date <= {snapshot_date}) as s
        
        on c.customerID = s.customerID
        
        inner join [SHUFFLE]
        
        (select * from core.products where productInService = 1) as p
        
        on t.ProductID = p.ProductID
        ) as f
        
        where t.next_txn_due_date >= {snapshot_date}
        """.format(**params)

        return_code, data = run_query(query, self.obj_spark)

        return return_code


    #generate labels for the customers above
    def _get_labels(self, params):

        query = """ 
        drop table if exists {db}.{user}_churn_events_{snapshot};
        create table {db}.{user}_churn_events_{snapshot}
        stored as parquet location "{data_loc}/{user}_churn_events_{snapshot}"
        
        as
        
        select a.customerID
        from
        (select a.customerID, 
        case when a.contract = "Month-to-month" then ADD_MONTHS(b.max_txn_date, 1)
        when a.contract = "One year" then ADD_MONTHS(b.max_txn_date, 12)
        when a.contract = "Two year" then ADD_MONTHS(b.max_txn_date, 24)
        else null end as next_txn_due_date,
        {snapshot_date} as snapshot_date
        from
        (select a.customerID, a.contract, b.max_txn_date
        from
        (select customerID, contract, transaction_date from core.transactions where transaction_date <= {snapshot_date}) as a
        left join [SHUFFLE]
        (select customerID, max(transaction_date) as max_txn_date from core.transactions where transaction_date <= {snapshot_date}) as b
        
        on 
        a.customerID = b.customerID
        and a.transaction_date = b.max_txn_date
        ) as c
        ) as d
        
        where datediff(next_txn_due_date, snapshot_date) > 31

        """.format(**params)

        return_code, data = run_query(query, self.obj_spark)

        return return_code

    #final table to serve
    def _get_snapshot(self, params):

        query = """ 
        drop table if exists {db}.{user}_churn_snapshot_{snapshot};
        create table {db}.{user}_churn_snapshot_{snapshot}
        stored as parquet location "{data_loc}/{user}_churn_snapshot_{snapshot}"
        
        as
        
        select straight_join a.* , case when b.customerID is null then "No" else "Yes" as Churn
        from
        {db}.{user}_churn_data_{snapshot} as a
        left join [SHUFFLE]
        {db}.{user}_churn_events_{snapshot} as b
        
        on a.customerID = b.customerID
        """.format(**params)

        return_code, data = run_query(query, self.obj_spark)

        return return_code
    
    def data_wrapper(self):
        
        return_dict = {
            'return_code':'',
            'return_msg':'',
        }
        
        params = self.get_params()
        
        return_code = self._get_data(params)
        if(return_code != 0):
            return_dict['return_code'] = return_code
            return_dict['return_msg'] = "Failed _get_data"
            return return_dict
        
        return_code = self._get_labels(params)
        if(return_code != 0):
            return_dict['return_code'] = return_code
            return_dict['return_msg'] = "Failed _get_labels"
            return return_dict
        
        return_code = self._get_snapshot(params)
        if(return_code != 0):
            return_dict['return_code'] = return_code
            return_dict['return_msg'] = "Failed _get_snapshot"
        
        return return_dict
    

