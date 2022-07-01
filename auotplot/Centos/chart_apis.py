from flask import Flask, request
from sqlalchemy import create_engine
from visualizer import recommender, plotter
@app.route("/getchart",methods=['POST'])
def getcharts():
    if request.method == 'POST':
        content =request.json
        print("333",content)
        client_id = content['client_id']
        with open('./files/db_data2.json') as f:
            dbjson = json.load(f)
        database = dbjson[client_id]['database']
        host = '203.112.158.76'
        username='root'
        password='Assisto@123'
        connection = "mysql+pymysql://" +username + ":"+password + "@"+ host + "/" + database
        engine = create_engine(connection,echo=True)
        tablename = content['tablename']
        select_column= content['select_columns']
        filter_column = content['filter_column']
        filter_value =  content['filter_value']
        appropriate_charts = content['appropriate_charts']
        query = test_query(select_column,tablename,filter_column,filter_value,client_id)
        #get query,connection and appropriate chart
        #return list of name
        chart = ["08_34_05stacked.png"] #this will the output
        data = {}
        data['images']=[]
        for cc in chart:
            data['images'].append("https://bot.delhicctv.in/images/"+cc)
            data['text']=[cc]
        return({"data":data})

@app.route("/readcharts",methods=['POST'])
def readcharts():
    if request.method == 'POST':
        content =request.json
        client_id = content['client_id']
        with open('./files/db_data2.json') as f:
            dbjson = json.load(f)
        database = dbjson[client_id]['database']
        host = '203.112.158.76'
        username='root'
        password='Assisto@123'
        connection = "mysql+pymysql://" +username + ":"+password + "@"+ host + "/" + database
        engine = create_engine(connection,echo=True)
        tablename = content['tablename']
        select_column = content['select_columns']
        filter_column = content['filter_column']
        filter_value =  content['filter_value']
        query = test_query(select_column,tablename,filter_column,filter_value,client_id)
        
        #take query and connection into visualizer return type of charts
        #stores in a list
        type_of_charts = ["Histogram","Barchart","PIE Chart"] #this will be ouput
        return({"data":type_of_charts})