import string
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


class Processing:

    def __init__(self, stopwords_path='data/', tokenizer_path='models/'):
        # It needs a stopwords file to init
        stop_words = pd.read_csv(stopwords_path + 'stopwords-es.txt', header=None)
        stop_words = stop_words[0].tolist() + ['secuela']
        self.stop_words = stop_words

        try:
            self.stemmer = SnowballStemmer("spanish", ignore_stopwords=True)
        except:
            nltk.download("popular")
            self.stemmer = SnowballStemmer("spanish", ignore_stopwords=True)

        # loading
        with open(tokenizer_path + 'tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.__vocab_size = len(self.tokenizer.word_index) + 1

    @property
    def vocab_size(self):
        return self.__vocab_size

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
        data['overview'] = data['overview'].apply(lambda x: self.clean_sentence(str(x)))
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

    # Split train_test
    def split_data(self, data):

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

    def tokenize_overview(self, overviews, max_len):

        X = self.tokenizer.texts_to_sequences(overviews)
        # print(len(max(X, key=len)))
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # We pad the sentence for the left to fit with max_len
        X = pad_sequences(X, padding='pre', maxlen=max_len)
        # print(X[1])

        return X

    def process(self, data, train):

        n_words = 12000
        max_len = 100

        df = self.clean_overview(data)
        df = self.paste_cast(df)

        if train:

            X_train, X_test, y_train, y_test = self.split_data(df)

            self.fit_tokenizer(X_train, n_words)
            X_train = self.tokenize_overview(X_train, max_len)
            X_test = self.tokenize_overview(X_test, max_len)

            return X_train, X_test

        else:

            X = df['overview'].values
            X = self.tokenize_overview(X, max_len)

            return X


