import argparse
import os
import ipfsapi
import pandas
import keras
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# tensorflow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser(description='Adds additional training to a model and outputs the hash of the finetuned model.')
parser.add_argument('--model', type=str, required=True, help='The hash of the original model')
parser.add_argument('--data', type=str, nargs='+', required=True, help='Filepath(s) of data to train the model with')

args = parser.parse_args()

api = ipfsapi.connect('127.0.0.1', 5001)


def finetune_model(original, x, y, **kwargs):
    model = Sequential()

    for layer in original.layers:
        model.add(layer)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    for layer in model.layers:
        layer.trainable = False

    model.fit(x, y, **kwargs)

    for layer in model.layers:
        layer.trainable = True

    return model

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

def save_model(model):
    model.save("tmp_model.dat")

    # save to ipfs
    saved_obj = api.add("tmp_model.dat")

    # delete temporary file
    os.remove("tmp_model.dat")

    return saved_obj

def train_with_csv_data(original, filepath):
    # TODO make this import another pre-trained model instead of training w/ data in here.
    print "Training with data from {}".format(filepath)

    df = pandas.read_csv(filepath, header=None)

    X = df.values[:, 0:4].astype(float)
    Y = df.values[:, 4]

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    finetune_model(original, X, dummy_y, epochs=200, batch_size=5, verbose=0, shuffle=False)


model = load_model()

for filepath in args.data:
    train_with_csv_data(model, filepath)

saved_obj = save_model(model)

print saved_obj['Hash']