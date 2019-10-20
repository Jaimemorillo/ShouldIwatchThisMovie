from like.preparation import Preparation
from like.processing import Processing
import numpy as np
import pandas as pd


class DBAppController:

    def __init__(self, data_path='data/'):

        self.pre = Preparation()
        # self.pro = Processing()
        self.path = data_path
        self.db_ini = self.load_ini_db()
        self.db_like_ini = self.load_like_db()
        self.db_like_act = pd.DataFrame(columns=['id', 'like'])
        self.db_act = self.db_ini[~self.db_ini.index.isin(self.db_like_ini.index)].copy()
        self.db_credits = self.load_credits_db()
        self.__sample = self.get_4_random_movies()
        self.db_train_data = None

    @property
    def sample(self):
        return self.__sample

    @sample.setter
    def sample(self, val):
        self.__sample = val

    def load_ini_db(self):

        movies_ini = self.pre.get_overview(self.path + 'tmdb_spanish_overview.csv')

        movies_ini['reduced_overview'] = movies_ini['overview'].apply(
            lambda x: " ".join(x.split()[0:30] + ['...']))
        movies_ini['prediction'] = 80
        movies_ini['like'] = np.nan

        movies_ini = movies_ini[['id', 'title', 'overview',
                                 'reduced_overview', 'prediction', 'like']]
        return movies_ini

    def load_like_db(self):

        like_ini = self.pre.get_personal_like(self.path + 'tmdb_spanish_Jaime2.csv')

        return like_ini

    def load_credits_db(self):

        credits_ini = self.pre.get_credits(self.path + 'tmdb_5000_credits.csv')

        return credits_ini

    def merge_train_data(self, batch):

        df_like_train = self.db_like_act.iloc[0:batch]
        df_like_lfo = self.db_like_act.iloc[batch:]
        data = self.pre.merge_over_like_credits(self.db_ini,
                                                df_like_train, self.db_credits)
        self.db_train_data = data

        print(data.columns)

        return df_like_lfo

    def get_4_random_movies(self):

        sample = self.db_act.sample(4)
        sample = self.pre.merge_over_credits(sample, self.db_credits)

        # Hacemos las predicciones
        # X = self.pro.process(sample)

        return sample

    def update_sample(self):

        m_with_like = self.sample[self.sample['like'].notna()]

        self.db_like_act = self.db_like_act.append(m_with_like[['id', 'like']])
        self.db_act = self.db_act.drop(m_with_like.index)

        batch = 4
        if len(self.db_like_act) >= batch:
            # Update db_like_act with leftover data over 16
            self.db_like_act = self.merge_train_data(batch)

        self.sample = self.get_4_random_movies()

        print(len(self.db_act))
        print(len(self.db_like_act))

        return None
