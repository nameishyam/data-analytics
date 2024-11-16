from flask import Flask, request, render_template
import pickle
import numpy as np

# ...existing code...
model = pickle.load(open('./model/model.pkl', 'rb'))
print(type(model))  # Add this line to check the type
# ...existing code...

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        hours_studied = int(request.form['hours_studied'])
        previous_scores = int(request.form['previous_scores'])
        activities = int(request.form['activities'])
        sleep_hours = int(request.form['sleep_hours'])
        sample_papers = int(request.form['sample_papers'])

        # Prepare input for prediction
        features = np.array([[hours_studied, previous_scores, activities, sleep_hours, sample_papers]])

        # Make prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='Predicted Performance Index: {}'.format(output))
    except Exception as e:
        return f'Error occurred: {str(e)}'

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000)