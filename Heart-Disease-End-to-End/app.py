import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('heart-disease-model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    """
    For rendering result on HTML GUI
    """
    int_features = [float(x) for x in request.form.values()]
    
    final_features = [np.array(int_features)]
    
    prediction = model.predict(final_features) 
    output = round(prediction[0],2)
    
    # if output ==1:
    #     output = "have heart disease"
    # if output ==0 :
    #     output = "dont have heart disease"
    # else : 
    #     output = " "
    
    print("hello")
    print(int_features)
    
    return render_template('index.html', prediction_text=f'you probably {output}')















if __name__ == "__main__":
    app.run(debug=True)