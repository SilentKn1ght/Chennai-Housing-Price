from flask import Flask, request, app, jsonify, url_for, render_template 
import pandas as pd
import numpy as np
import joblib
import json
from babel.numbers import format_currency

app=Flask(__name__)
#load model and scaler
model_pipe=joblib.load(open('Model_Pipe.sav','rb'))
model=joblib.load(open('ChennaiHousepricing.sav','rb'))
scaler=joblib.load(open('scaler.sav','rb'))
df= pd.read_csv("Cleaned_data.csv") 

@app.route('/')
def home():
    
    locations = sorted(df['AREA'].unique())
    sales = sorted(df['SALE_COND'].unique())
    parkings = sorted(df['PARK_FACIL'].unique())
    buildtypes = sorted(df['BUILDTYPE'].unique())
    utilities = sorted(df['UTILITY_AVAIL'].unique())
    streets = sorted(df['STREET'].unique())
    zones = sorted(df['MZZONE'].unique())
    return render_template('home.html', locations=locations, sales=sales, parkings=parkings, 
                           buildtypes=buildtypes, utilities=utilities, streets=streets, zones=zones)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    total_sqft= int(request.form.get('total_sqft'))
    dist= int(request.form.get('dist'))
    bedroom= int(request.form.get('bedroom'))
    bathroom= int(request.form.get('bathroom'))
    room= int(request.form.get('room'))
    sale_cond= request.form.get('sale_cond')
    parking= request.form.get('parking')
    Buildtype= request.form.get('Buildtype')
    utility= request.form.get('utility')
    street= request.form.get('street')
    zone= request.form.get('zone')
    House_age= int(request.form.get('House_age'))
    
    print(location, total_sqft,dist,bedroom,bathroom,room,sale_cond,parking,Buildtype,utility,street,zone,House_age)
    input=pd.DataFrame([[location, total_sqft,dist,bedroom,bathroom,room,sale_cond,parking,Buildtype,utility,street,zone,House_age]], 
                       columns=['AREA','INT_SQFT','DIST_MAINROAD','N_BEDROOM','N_BATHROOM','N_ROOM','SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL','STREET','MZZONE','HOUSE_AGE'])
    
    output=model_pipe.predict(input)[0]
    print(np.round(output, 2))
    return render_template("home.html",prediction_text="The Predicted House Price is : {}".format(format_currency(output, 'INR', locale='en_IN')))

    
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)[0]
    print(output)
    return jsonify(format_currency(output, 'INR', locale='en_IN'))

if __name__=='__main__':
    app.run(debug=True)