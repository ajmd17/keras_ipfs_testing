import argparse
import os
import sys
import ipfsapi
import pandas

# tensorflow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser(description='Uses a trained model stored on ipfs to make predictions.')
parser.add_argument('--model', type=str, required=True, help='The hash of the model to use')
parser.add_argument('--input', type=str, required=True, help='Filepath of input data (read from stdout if not provided)')

args = parser.parse_args()

import keras
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

api = ipfsapi.connect('127.0.0.1', 5001)

def load_model_by_hash(model_hash):
    # download the model
    api.get(model_hash)

    # load the model in
    loaded_model = keras.models.load_model(model_hash)

    return loaded_model

def load_model():
    try:
        model = load_model_by_hash(args.model)

        if model is None:
            raise Exception("None was returned")

        return model

    except Exception as e:
        print "Failed to load model {}: {}".format(args.model, e.message)
        exit(1)


model = load_model()

print "model = {}".format(model)