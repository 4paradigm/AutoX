from flask import Flask
from flask_restful import reqparse, Api, Resource
import pickle

app = Flask(__name__)
api = Api(app)

def load_obj(name):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)

recall_and_rank = load_obj('recall_and_rank')

popular_item = ['[44]',
                 '[02]',
                 '[21]',
                 '[24]',
                 '[05]',
                 '[27]',
                 '[31]',
                 '[03]',
                 '[23]',
                 '[99]',
                 '[22]',
                 '[74]']

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('uid')

class PredictSentiment(Resource):
    def get(self):

        args = parser.parse_args()
        uid = args['uid']

        if uid in recall_and_rank:
            cur_res = recall_and_rank[uid][0][:12]
            cur_res = [x[:4] for x in cur_res]
        else:
            cur_res = popular_item

        output = {'uid': uid, 'predicition': cur_res}
        return output

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=False)
