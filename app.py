from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('Customer_Churn.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting data from the form
        credit_score = float(request.form['credit_score'])
        gender = request.form['gender']
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_of_products = int(request.form['num_of_products'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])
        estimated_salary = float(request.form['estimated_salary'])

        # Preprocessing input (manual handling of Gender)
        gender_map = {'Male': 1, 'Female': 0}

        # Create input feature vector in the same format as model was trained
        input_features = [
            credit_score,
            gender_map[gender],
            age,
            tenure,
            balance,
            num_of_products,
            has_cr_card,
            is_active_member,
            estimated_salary
        ]

        # Convert the input to the numpy array
        input_features = np.array([input_features])

        # Predict using the model
        prediction = model.predict(input_features)

        # Return the prediction to the user
        result = 'Customer will churn' if prediction[0] == 1 else 'Customer will not churn'

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
