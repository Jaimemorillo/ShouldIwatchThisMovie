from like.preparation import Preparation
from like.processing import Processing
from like.modelling import Modelling
import numpy as np
import pandas as pd
import keras.backend.tensorflow_backend as tb


class DBRandomController:

    def __init__(self, data_path='data/', models_path='models/'):

        self.pre = Preparation()
        self.pro = Processing(stopwords_path=data_path, tokenizer_path=models_path, max_len=80)
        self.mod = Modelling(vocab_size=self.pro.vocab_size, model_path=models_path, max_len=80)
        self.path = data_path
        self.db_ini = self.load_ini_db()
        self.db_like_ini = self.load_like_db()
        self.db_like_act = pd.DataFrame(columns=['id', 'like'])
        self.db_act = self.db_ini[~self.db_ini.index.isin(self.db_like_ini.index)].copy()
        self.__sample = self.get_4_random_movies()

    @property
    def sample(self):
        return self.__sample

    @sample.setter
    def sample(self, val):
        self.__sample = val

    def load_ini_db(self):

        movies_ini = self.pre.get_overview(self.path + 'tmdb_spanish_def.csv')

        movies_ini['reduced_overview'] = movies_ini['overview'].apply(
            lambda x: " ".join(x[0:150].split()[0:-1]) + '...')
        movies_ini['prediction'] = 80
        movies_ini['like'] = np.nan

        movies_ini = movies_ini[['id', 'title', 'overview',
                                 'reduced_overview', 'prediction',
                                 'cast', 'crew', 'like']]
        return movies_ini

    def load_like_db(self):

        like_ini = self.pre.get_personal_like(self.path + 'tmdb_spanish_Jaime_def.csv')

        return like_ini

    def merge_train_data(self, batch):

        df_like_train = self.db_like_act.iloc[0:batch]
        df_like_left = self.db_like_act.iloc[batch:]
        df_train = self.pre.merge_over_like(self.db_ini, df_like_train)

        return df_train, df_like_left

    def get_4_random_movies(self):

        tb._SYMBOLIC_SCOPE.value = True
        sample = self.db_act.sample(4)

        # Hacemos las predicciones
        X = self.pro.process(data=sample.copy(), train_dev=False)
        pred, score = self.mod.predict(X)
        score = [int(round(s[0]*100)) for s in score]
        sample['prediction'] = score

        return sample

    def update_sample(self):

        m_with_like = self.sample[self.sample['like'].notna()]

        self.db_like_act = self.db_like_act.append(m_with_like[['id', 'like']])
        self.db_act = self.db_act.drop(m_with_like.index)

        batch = 4
        if len(self.db_like_act) >= batch:

            train, left = self.merge_train_data(batch)
            self.db_like_act = left

            X_train = self.pro.process(data=train.copy(), train_dev=False)
            y_train = train.like.values
            tb._SYMBOLIC_SCOPE.value = True
            self.mod.fit_model(X_train, y_train, epochs=1, batch_size=4)

        self.sample = self.get_4_random_movies()

        print(len(self.db_act))
        print(len(self.db_like_act))

        return None
