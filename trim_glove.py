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

MIN_JOKE_LENGTH = 10
MAX_JOKE_LENGTH = 40

joke_vocabulary = {}

training_jokes = {}
test_jokes = {}
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

                if filename == joke_paths[0]:
                    test_jokes[joke_id] = [joke_body, joke_rating]
                else:
                    if np.random.rand() > 0.7:
                        test_jokes[joke_id] = [joke_body, joke_rating]
                    else:
                        training_jokes[joke_id] = [joke_body, joke_rating]


gloveInitFile = "word-embeddings/glove.42B.300d.txt"
gloveModFile  = "word-embeddings/glove.mod.300d.txt"
f = open(gloveInitFile,'r')
w = open(gloveModFile, 'w')
for i, line in enumerate(f):
    splitLine = line.split()
    word = splitLine[0]
    if joke_vocabulary.get(word) != None or word == '...':
        w.write(line)
    if i%100000 == 0:
        print('read in %d words', i)
