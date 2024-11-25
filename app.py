from flask import Flask, request, render_template
import pickle
import numpy as np

model = pickle.load(open('./model/model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours_studied = int(request.form['hours_studied'])
        previous_scores = int(request.form['previous_scores'])
        activities = int(request.form['activities'])
        sleep_hours = int(request.form['sleep_hours'])
        sample_papers = int(request.form['sample_papers'])

        features = np.array([[hours_studied, previous_scores, activities, sleep_hours, sample_papers]])

        prediction = model.predict(features)
        prediction = np.round(prediction, decimals=2)
        prediction = np.clip(prediction, 0, 100)
        if (prediction[0] >= 35):
            result = 'May be pass'
        else:
            result = 'May be fail'
        output = f'{prediction[0]} ({result})'

        return render_template('index.html', prediction_text='Predicted Performance Index: {}'.format(output, result))
    except Exception as e:
        return f'Error occurred: {str(e)}'

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000)