
import re
import json
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import text_to_word_sequence

joke_paths = ["joke-dataset/wocka.json", "joke-dataset/stupidstuff.json", "joke-dataset/reddit_jokes.json"]
non_joke_paths = ["sentence-dataset/sentences.csv"]
WORD_EMBEDDINGS_PATH = 'word-embeddings/saved_glovedata.json'

MIN_JOKE_LENGTH = 10
MAX_JOKE_LENGTH = 40


joke_vocabulary = {}

def loadJokes():
    training_jokes = []
    test_jokes = []
    i = 0
    for filename in sorted(joke_paths):
        with open(filename) as f:
            joke_data = json.load(f)
            for joke in joke_data:

                if filename == joke_paths[0]:
                    joke_rating = -1
                    joke_id = joke["id"]
                    joke_body = joke["body"]

                elif filename == joke_paths[1]:
                    joke_rating = float(joke["rating"]) * 2
                    joke_id = joke["id"]
                    joke_body = joke["body"]

                elif filename == joke_paths[2]:
                    joke_body = joke["title"] + joke["body"]
                    joke_id = joke["id"]
                    score = float(joke["score"]) / 10
                    joke_rating = min([score, 10])

                else:
                    print("error?")

                joke_body = text_to_word_sequence(joke_body)
                joke_length =  len(joke_body)
                if joke_length < MAX_JOKE_LENGTH and joke_length > MIN_JOKE_LENGTH:
                    for word in joke_body:
                        if joke_vocabulary.get(word) == None:
                            joke_vocabulary[word] = i
                            i = i + 1
                    if joke_rating == 0:
                        if np.random.rand() > .01:
                            joke_rating = np.random.rand() * 10
                    elif joke_rating < 1.5:
                        joke_rating += np.random.rand() * 5
                    if filename == joke_paths[0]:
                        test_jokes.append([joke_body, joke_rating])
                    else:
                        if np.random.rand() > 0.7:
                            test_jokes.append([joke_body, joke_rating])
                        else:

                            training_jokes.append([joke_body, joke_rating])
    return [training_jokes, test_jokes]

def nonJokes():
    non_jokes = []
    for filename in sorted(non_joke_paths):
        with open(filename) as f:
            reader = csv.reader(f)
            for line in reader:
                if len(str(line)) > 45:
                    non_jokes.append(line)
    return non_jokes

def loadGloveModel():
    try: # letting it just load each time until I find a better way to save my dict of np arrays
        print("Trying to load pre-saved Glove Model")
        with open(WORD_EMBEDDINGS_PATH, 'r') as fp:
            model = json.load(fp)
            return model
    except:
        print("Reloading Glove Model from Scratch...")
        gloveFile = "word-embeddings/glove.mod.300d.txt"
        f = open(gloveFile,'r')
        model = {}
        for i, line in enumerate(f):
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array(splitLine[1:], dtype=float)
            model[word] = embedding
            if i%100000 == 0:
                print('read in %d words', i)
       # with open(WORD_EMBEDDINGS_PATH, 'w') as fp:
       #     json.dump(model, fp)
       # print("Saved ", len(model), " words to file system!")
        print("Done. ", len(model), " words loaded!")
        return model

[training_jokes, test_jokes] = loadJokes()
#non_jokes = nonJokes()

print("we have ", len(training_jokes), " training jokes")
print("we have ", len(test_jokes), " test jokes")
#print("we have ", len(non_jokes), " non-jokes")

# print(len(joke_vocabulary)) = 248676, log base 2 < 18

#for word in joke_vocabulary:            # only adding words that I have encodings for, otherwise just skipping them lol - probably best fixed with preprocessing

#    joke_vocabulary[word] = list('{0:018b}'.format(8))

joke_vocabulary = loadGloveModel()

training_joke_bodies = []
training_joke_ratings = []
for joke_info in training_jokes:
    training_joke_ratings.append(joke_info[1])
    embedded_joke = []
    for word in joke_info[0]:
        try:
            encoding = joke_vocabulary[word]
            embedded_joke.append(encoding)
        except KeyError:
            pass
    while len(embedded_joke) < MAX_JOKE_LENGTH:
        embedded_joke.append(joke_vocabulary['...'])
    embedded_joke = np.array(embedded_joke)
    training_joke_bodies.append(embedded_joke)
training_joke_bodies = np.array(training_joke_bodies)
training_joke_ratings = np.array(training_joke_ratings)#.reshape((len(training_joke_ratings), 1, 1))

test_joke_bodies = []
test_joke_ratings = []
for joke_info in test_jokes:
    test_joke_ratings.append(joke_info[1])
    embedded_joke = []
    for word in joke_info[0]:
        try:
            # only adding words that I have encodings for, otherwise just skipping them lol - probably best fixed with preprocessing
            encoding = joke_vocabulary[word]
            embedded_joke.append(encoding)
        except KeyError:
            pass
    while len(embedded_joke) < MAX_JOKE_LENGTH:
        embedded_joke.append(joke_vocabulary['...'])
    embedded_joke = np.array(embedded_joke)
    test_joke_bodies.append(embedded_joke)
test_joke_bodies = np.array(test_joke_bodies)
test_joke_ratings = np.array(test_joke_ratings)#.reshape((len(test_joke_ratings), 1, 1))

print('hi')
print(training_joke_bodies.shape)
print(training_joke_ratings.shape)
print(test_joke_bodies.shape)
print(test_joke_ratings.shape)
print(training_joke_bodies[0].shape, training_joke_bodies[5].shape, training_joke_bodies[30].shape)


print('0:  ', len(training_joke_ratings[(training_joke_ratings == 0)]))
print('0-1:  ', len(training_joke_ratings[( training_joke_ratings > 0 ) & (training_joke_ratings < 1)]))
print('1-2:  ', len(training_joke_ratings[( training_joke_ratings >= 1 ) & (training_joke_ratings < 2)]))
print('2-3:  ', len(training_joke_ratings[( training_joke_ratings >= 2 ) & (training_joke_ratings < 3)]))
print('3-4:  ', len(training_joke_ratings[( training_joke_ratings >= 3 ) & (training_joke_ratings < 4)]))
print('4-5:  ', len(training_joke_ratings[( training_joke_ratings >= 4 ) & (training_joke_ratings < 5)]))
print('5-6:  ', len(training_joke_ratings[( training_joke_ratings >= 5 ) & (training_joke_ratings < 6)]))
print('6-7:  ', len(training_joke_ratings[( training_joke_ratings >= 6 ) & (training_joke_ratings < 7)]))
print('7-8:  ', len(training_joke_ratings[( training_joke_ratings >= 7 ) & (training_joke_ratings < 8)]))
print('8-9:  ', len(training_joke_ratings[( training_joke_ratings >= 8 ) & (training_joke_ratings < 9)]))
print('9-10:  ', len(training_joke_ratings[(training_joke_ratings >= 9)]))
#"""

in_out_neurons = (None, 300)
hidden_neurons = 500

model = Sequential()
#model.add(LSTM(hidden_neurons, input_shape=in_out_neurons, return_sequences=True))
model.add(LSTM(hidden_neurons, input_shape=(40, 300)))
#model.add(LSTM(hidden_neurons, return_sequences=False))
model.add(Dense(1))
#model.add(Activation("relu"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")



#(training_joke_bodies, training_joke_ratings), (test_joke_bodies, test_joke_ratings) = train_test_split(data)  # retrieve data
model.fit(training_joke_bodies, training_joke_ratings, batch_size=700, epochs=10, validation_split=0.05)

predicted = model.predict(test_joke_bodies)
#rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

with open('results.txt', 'w') as f:
    for i, line in enumerate(predicted):
        original_joke = test_jokes[i]
        joke_text = original_joke[0]
        original_rating = original_joke[1]
        f.write('Joke: {}\nTheir Rating: {} ::: Our Rating: {}\n\n'.format(joke_text, original_rating, line))


# and maybe plot it
#pd.DataFrame(predicted).to_csv("predicted.csv")
#pd.DataFrame(test_joke_ratings).to_csv("test_data.csv")

#"""


"""
import pandas as pd
from random import random

flow = (list(range(1,10,1)) + list(range(10,1,-1)))*100
pdata = pd.DataFrame({"a":flow, "b":flow})
pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:] * random()  # some noise

import numpy as np

def _load_data(data, n_prev = 100):
    #data should be pd.DataFrame()

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    #This just splits data to training and testing parts
    
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 2
hidden_neurons = 50

model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("relu"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
model.fit(X_train, y_train, batch_size=700, epochs=10, validation_split=0.05)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

# and maybe plot it
pd.DataFrame(predicted).to_csv("predicted.csv")
pd.DataFrame(y_test).to_csv("test_data.csv")
"""