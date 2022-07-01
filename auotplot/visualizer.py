import pandas as pd
from sqlalchemy import create_engine
from dateutil.parser import parse
import json
from dataprofiler import Data, Profiler
from autoplot import profiler
#engine = create_engine("mysql+mysqlconnector://root:sh90j9999@localhost/flaskapp", echo=True)

#query='select `Active Screen Time`, `Idle Time`, `Productive Screentime`, `UnProductive Screentime` from analysis;'
def plotter(query,engine):
    dfan=pd.read_sql(query, con=engine)
    dtype_num=['int64','float64']
    dtype_o=['object']


    c_count=0
    c_name=[]
    n_count=0
    n_name=[]
    count_json={}
    for i in dfan.columns:
        if dfan[i].dtype in dtype_num:
            n_count+=1
            n_name.append(i)
        elif dfan[i].dtype in dtype_o:
            c_count+=1
            c_name.append(i)
    profile = Profiler(dfan) # Calculate Statistics, Entity Recognition, etc
    readable_report = profile.report(report_options={"output_format": "compact"})
    data_stats=readable_report['data_stats']
    count_json['no_column']=dfan.shape[1]

    for i in data_stats:
        if i['categorical']==True:
            c_count+=1
            c_name.append(i['column_name'])
        else:
            n_count+=1
            n_name.append(i['column_name'])
    if c_count and c_name:
        count_json['c_count']=c_count
        count_json['c_name']=c_name
    else:
        count_json['c_count']=False
        count_json['c_name']=False

    if n_count and n_name:
        count_json['n_name']=n_name
        count_json['n_count']=n_count
    else:
        count_json['n_name']=False
        count_json['n_count']=False


    rtn=profiler(dfan,count_json) 
    return rtn


if __name__=='__main__':
    engine = create_engine("mysql+mysqlconnector://root:sh00j0000@localhost/flaskapp", echo=True)
    query='select `Active Screen Time` from analysis where `Agent Name`="MCARE-AAKASH SIKKA";'
    rt=plotter(query,engine)
    