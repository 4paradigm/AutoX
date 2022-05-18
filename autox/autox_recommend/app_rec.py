from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np

import sys

sys.path.append('../AutoX')
from autox import AutoXRecommend

app = Flask(__name__)
api = Api(app)

autoXRecommend = AutoXRecommend()

path = 'autoXRecommend_popular_temp'
autoXRecommend.load(path, mode='recalls', recall_method='popular')

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('uids')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        uids = args['uids']
        uids = uids.split(' ')
        print(uids)

        res = autoXRecommend.transform(uids)
        print(res)

        #
        # # vectorize the user's query and make a prediction
        # uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        # prediction = model.predict(uq_vectorized)
        # pred_proba = model.predict_proba(uq_vectorized)
        #
        # # Output either 'Negative' or 'Positive' along with the score
        # if prediction == 0:
        #     pred_text = 'Negative'
        # else:
        #     pred_text = 'Positive'
        #
        # # round the predict proba value and set to new variable
        # confidence = round(pred_proba[0], 3)
        #
        # # create JSON object
        # output = {'prediction': pred_text, 'confidence': confidence}

        return res.to_json(orient="index")


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=False)
