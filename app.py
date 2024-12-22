from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model/pcos_svm_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')  # Render the test.html file

@app.route("/result")
def result():
    return render_template("result.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect input data
            input_data = [
                int(request.form['Age']),                 # Age (yrs)
                float(request.form['Weight']),           # Weight (Kg)
                float(request.form['Height']),           # Height (Cm)
                float(request.form['BMI']),              # BMI
                int(request.form['Blood_Group']),        # Blood Group (Mapped values: A+ -> 11, O- -> 16, etc.)
                int(request.form['Cycle']),              # Cycle (R -> 2, I -> 5)
                int(request.form['Cycle_length']),       # Cycle length (days)
                int(request.form['Marriage_Status']),    # Marriage Status (Yrs)
                int(request.form['Pregnant']),           # Pregnant (Y -> 1, N -> 0)
                int(request.form['No_of_abortions']),    # No. of abortions
                int(request.form['Weight_gain']),        # Weight gain (Y -> 1, N -> 0)
                int(request.form['Hair_growth']),        # Hair growth (Y -> 1, N -> 0)
                int(request.form['Skin_darkening']),     # Skin darkening (Y -> 1, N -> 0)
                int(request.form['Hair_loss']),          # Hair loss (Y -> 1, N -> 0)
                int(request.form['Pimples']),            # Pimples (Y -> 1, N -> 0)
                int(request.form['Fast_food']),          # Fast food (Y -> 1, N -> 0)
                int(request.form['Reg_Exercise'])        # Regular Exercise (Y -> 1, N -> 0)
            ]

            # Convert to numpy array and scale
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)

            result = {
                "prediction": "PCOS" if prediction == 1 else "No PCOS",
                "probability_PCOS": f"{prediction_proba[0][1] * 100:.2f}%",
                "probability_No_PCOS": f"{prediction_proba[0][0] * 100:.2f}%"
            }

            return render_template('result.html', result=result)

        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
