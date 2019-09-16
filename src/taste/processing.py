import string
import pandas as pd
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer


class Processing:

    def __init__(self, stopwords_path, tokenizer_path):
        # It needs a stopwords file to init
        stop_words = pd.read_csv(stopwords_path, header=None)
        stop_words = stop_words[0].tolist() + ['secuela']
        self.stop_words = stop_words
        self.stemmer = SnowballStemmer("spanish", ignore_stopwords=True)
        self.tokenizer = None
        self.vocab_size = None

    def normalize(self, s):
        replacements = (
            ("á", "a"),
            ("é", "e"),
            ("í", "i"),
            ("ó", "o"),
            ("ú", "u"),
        )
        for a, b in replacements:
            s = s.replace(a, b).replace(a.upper(), b.upper())

        return s

    def clean_sentence(self, x):
        # Clean sentence from punctuation, numbers and make it lower case
        x = self.normalize(x.lower())
        x = x.translate(str.maketrans('', '', string.punctuation))
        x = x.translate(str.maketrans('', '', '1234567890ªº'))

        return x

    def delete_stop_words(self, x):
        # Clean sentence from stopwords
        words = x.split(' ')
        words = [word for word in words if word not in self.stop_words]
        x = str(' '.join(words))

        return x

    def stem_sentence(self, sentence):
        # Stem the sentence
        stemmed_text = [self.stemmer.stem(word) for word in word_tokenize(sentence)]

        return " ".join(stemmed_text)

    def clean_overview(self, data):
        # Execute the full cleaning process into every overview
        data['overview'] = data['overview'].apply(lambda x: self.clean_overview(str(x)))
        data['overview'] = data['overview'].apply(lambda x: self.delete_stop_words(x))
        data['overview'] = data['overview'].apply(lambda x: self.stem_sentence(x))
        data['overview'] = data['overview'].apply(lambda x: self.delete_stop_words(x))

        return data

    # Get staff and paste to overview

    def get_actors(self, cast):

        try:

            json_cast = json.loads(cast)

        except:

            json_cast = cast

        if len(json_cast) > 2:
            up = 3
        else:
            up = len(json_cast)

        actors = ''

        for i in range(0, up):
            actor = json_cast[i]['name']
            actor = self.normalize(actor.replace(' ', '_').lower())

            actors = actors + ' ' + actor

        return actors

    def get_director(self, crew):

        try:

            json_crew = json.loads(crew)

        except:

            json_crew = crew

        directors = [member['name'] for member in json_crew if member['job'] == 'Director']
        directors = [self.normalize(director.replace(' ', '_').lower()) for director in directors]
        directors = str(' '.join(directors))

        return directors

    def paste_cast(self, data):

        data['overview'] = data.apply(lambda x: self.get_actors(x['cast']) + ' ' + x['overview'], axis=1)
        data['overview'] = data.apply(lambda x: self.get_director(x['crew']) + x['overview'], axis=1)

        return data

    def split_data(self, data ):

        overviews = data['overview'].values
        y = data['like'].values

        overviews_train, overviews_test, y_train, y_test = train_test_split(overviews, y, test_size=0.15, stratify=y,
                                                                            random_state=9)

        return overviews_train, overviews_test, y_train, y_test

    def fit_tokenizer(self, overviews_train, num_words):
        self.tokenizer = Tokenizer(num_words)
        self.tokenizer.fit_on_texts(overviews_train)
        # Adding 1 because of reserved 0 index
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def tokenize_overview(self, overviews, maxlen):

        X = self.tokenizer.texts_to_sequences(overviews)
        print(len(max(X, key=len)))
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # We pad the sentence for the left to fit with maxlen
        X = pad_sequences(X, padding='pre', maxlen=maxlen)
        print(X[1])

        return X