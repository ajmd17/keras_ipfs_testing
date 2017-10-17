import numpy
import pandas
import random
import keras

numpy.random.seed(7)

from math import floor, ceil
from keras.models import Sequential, Model
from keras.layers import Dense, Add, Merge, merge, AveragePooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split


df = pandas.read_csv("Iris.csv", header=None)
dataset = df.values

X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X_arr = X[:]
Y_arr = dummy_y[:]

random.shuffle(X_arr)
random.shuffle(Y_arr)

def do_model1():
    X_model1 = X_arr[int(len(X) / 2):]
    Y_model1 = Y_arr[int(len(dummy_y) / 2):]

    model = baseline_model()
    model.fit(X_model1, Y_model1, epochs=200, batch_size=5, verbose=0, shuffle=False)
    return model

def do_model2():
    X_model2 = X_arr[:int(len(X) / 2)]
    Y_model2 = Y_arr[:int(len(dummy_y) / 2)]

    model = baseline_model()
    model.fit(X_model2, Y_model2, epochs=200, batch_size=5, verbose=0, shuffle=False)
    return model

model1 = do_model1()
model2 = do_model2()

# model_merged = baseline_model()
# # load model1 weights
# model_merged.load_weights("model1_weights.dat")

# for layer in model_merged.layers:
#     layer.trainable = False

# # train on model2 data
# X_model2 = X_arr[int(len(X) / 2):]
# Y_model2 = Y_arr[int(len(dummy_y) / 2):]
# model_merged.fit(X_model2, Y_model2, epochs=200, batch_size=5, verbose=0, shuffle=False)

def finetune_model(original, x, y, **kwargs):
    merged = Sequential()

    for layer in original.layers:
        merged.add(layer)

    merged.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    for layer in merged.layers:
        layer.trainable = False

    merged.fit(x, y, **kwargs)

    for layer in merged.layers:
        layer.trainable = True

    return merged


model1_copy = baseline_model()
model1_copy.set_weights(model1.get_weights())
model_merged = finetune_model(model1_copy, X_arr[:int(len(X) / 2)], Y_arr[:int(len(dummy_y) / 2)], epochs=200, batch_size=5, verbose=0, shuffle=False)
model_merged.summary()



print "Doing some random predictions..."

for i in range(0, 25):
    a,b,c,d = [round(X_arr[i % len(X_arr), j], 2) for j in range(0, 4)]

    print "Testing with {}...".format((a, b, c, d))

    proba = model1.predict_proba(numpy.array([[a, b, c, d]]))[0]
    print "Model #1: ", encoder.inverse_transform([proba.argmax()]), "({})".format(proba) #encoder.inverse_transform(model1.predict(numpy.array([[a, b, c, d]])))

    proba = model2.predict_proba(numpy.array([[a, b, c, d]]))[0]
    print "Model #2: ", encoder.inverse_transform([proba.argmax()]), "({})".format(proba)

    proba = model_merged.predict_proba(numpy.array([[a, b, c, d]]))[0]
    print "Model (merged): ", encoder.inverse_transform([proba.argmax()]), "({})".format(proba)

    print "\n"