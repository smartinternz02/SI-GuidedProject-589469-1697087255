from flask import Flask, render_template, url_for, request,send_from_directory
import joblib

app = Flask(__name__)

model = joblib.load('Kiva_classification_RF.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Loan_Amount = request.form['loan'] # requesting loan amount
    Term = request.form['term'] # requesting for terms of loan
    lender_count = request.form['Country'] # requesting lender count
    Male = request.form['Male'] # requesting male count
    Female = request.form['Female'] # requesting female count
    X = [[int(Loan_Amount),int(Term),int(lender_count),int(Male),int(Female)]]
    prediction = model.predict(X) 
    print(prediction)
    
    return render_template("index.html",prediction='Loan Type  {}'.format(prediction[0]))
if __name__ == "__main__":
    app.run(debug=False)
