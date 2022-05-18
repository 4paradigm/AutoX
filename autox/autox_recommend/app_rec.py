from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd

import sys

app = Flask(__name__)
api = Api(app)


path = 'res.csv'

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('uid')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        uid = args['uid']

        recall_and_rank = pd.read_csv(path)
        popular_item = ['a', 'b']

        cur_res = recall_and_rank.loc[recall_and_rank['uid'] == uid, 'prediction']
        if cur_res.shape[0] == 1:
            return cur_res
        else:
            return popular_item

        # # create JSON object
        # output = {'prediction': pred_text, 'confidence': confidence}

        return res.to_json(orient="index")


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=False)
