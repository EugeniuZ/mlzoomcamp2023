import os
import pickle

from flask import Flask
from flask import request
from flask import jsonify


MODEL = None
DV = None


with open(os.getenv('MODEL_PATH', 'model1.bin'), 'rb') as f:
    MODEL = pickle.load(f)


with open(os.getenv('DV_PATH', 'dv.bin'), 'rb') as f:
    DV = pickle.load(f)


app = Flask('credit')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    print(customer)

    X = DV.transform([customer])
    y_pred = MODEL.predict_proba(X)[0, 1]
    credit = y_pred >= 0.5

    result = {
        'credit_probability': float(y_pred),
        'credit': bool(credit)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9991)
