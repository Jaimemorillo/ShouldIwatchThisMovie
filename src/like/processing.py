import string
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import pickle
import re
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf


class Processing:

    def __init__(self, stopwords_path='data/', tokenizer_path='models/'):
        # It needs a stopwords file to init
        stop_words = pd.read_csv(stopwords_path + 'stopwords-es.txt', header=None)
        stop_words = stop_words[0].tolist() + ['secuela']
        self.stop_words = stop_words
        self.n_words = 8000
        self.max_len = 80
        # self.aug = naf.Sequential([
        #    naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-cased', action="insert", aug_p=0.1),
        #    naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-cased', action="substitute", aug_p=0.9),
        #    naw.RandomWordAug(action="delete", aug_p=0.1)
        # ])

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
        s = s.lower()
        replacements = (
            ("á", "a"),
            ("é", "e"),
            ("í", "i"),
            ("ó", "o"),
            ("ú", "u"),
            ("ñ", "n")
        )
        for a, b in replacements:
            s = s.replace(a, b).replace(a.upper(), b.upper())

        return s

    def split_punt(self, x):
        words = WordPunctTokenizer().tokenize(x)
        x = str(' '.join(words))
        x = re.sub(' +', ' ', x)

        return x

    def delete_stop_words(self, x):
        x = x.translate(str.maketrans('', '', string.punctuation))
        x = x.translate(str.maketrans('', '', '1234567890ªº¡¿'))
        words = x.split(' ')
        words = [word for word in words if word not in self.stop_words]
        x = str(' '.join(words))

        return x

    def stem_sentence(self, sentence):
        # Stem the sentence
        stemmed_text = [self.stemmer.stem(word) for word in word_tokenize(sentence)]

        return " ".join(stemmed_text)

    def augment(self, x):
        try:
            return self.aug.augment(x)
        except:
            return None

    def clean_overview(self, df):
        # Execute the full cleaning process into every overview
        df['overview'] = df['overview'].apply(lambda x: self.normalize(x))
        df['overview'] = df['overview'].apply(lambda x: self.delete_stop_words(x))
        df['overview'] = df['overview'].apply(lambda x: self.stem_sentence(x))
        df['overview'] = df.apply(lambda x: self.get_actors(x['cast']) + ' ' + x['overview'], axis=1)
        df['overview'] = df.apply(lambda x: self.get_director(x['crew']) + x['overview'], axis=1)
        df['overview'] = df['overview'].apply(lambda x: self.normalize(x))
        df['overview'] = df['overview'].apply(lambda x: self.delete_stop_words(x))

        return df

    # Get staff and paste to overview
    @staticmethod
    def eval_cell(cell):

        try:

            cell_array = eval(cell)

        except:

            cell_array = []

        return cell_array

    def get_actors(self, cast):

        eval_cast = self.eval_cell(cast)

        if len(eval_cast) > 2:
            up = 3
        else:
            up = len(eval_cast)

        actors = ''

        for i in range(0, up):
            actor = eval_cast[i]['name']
            actor = self.normalize(actor.replace(' ', '_').lower())

            actors = actors + ' ' + actor

        return actors

    def get_director(self, crew):

        eval_crew = self.eval_cell(crew)

        directors = [member['name'] for member in eval_crew if member['job'] == 'Director']
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
        from keras.preprocessing.sequence import pad_sequences

        # We pad the sentence for the left to fit with max_len
        X = pad_sequences(X, padding='pre', maxlen=max_len)
        # print(X[1])

        return X

    def process(self, data, train_dev):

        df = self.clean_overview(data)
        df = self.paste_cast(df)

        if train_dev:

            X_train, X_test, y_train, y_test = self.split_data(df)

            self.fit_tokenizer(X_train, self.n_words)
            X_train = self.tokenize_overview(X_train, self.max_len)
            X_test = self.tokenize_overview(X_test, self.max_len)

            return X_train, X_test

        else:

            X = df['overview'].values
            X = self.tokenize_overview(X, self.max_len)

            return X


