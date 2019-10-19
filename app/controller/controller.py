from taste.preparation import Preparation
import numpy as np


class DBController:

    def __init__(self, data_path='data/'):
        self.pre = Preparation()
        self.path = data_path
        self.__db_ini = self.load_ini_db()
        self.__db_like = self.load_like_db()
        self.__db_act = self.db_ini[~self.db_ini.index.isin(self.db_like.index)].copy()
        self.__sample = self.get_4_random_movies()
        self.__db_credits = self.load_credits_db()
        self.__db_full_data = None

    @property
    def db_ini(self):
        return self.__db_ini

    @property
    def db_act(self):
        return self.__db_act

    @property
    def db_like(self):
        return self.__db_like

    @property
    def db_credits(self):
        return self.__db_credits

    @property
    def sample(self):
        return self.__sample

    @property
    def db_full_data(self):
        return self.__db_full_data

    @db_ini.setter
    def db_ini(self, val):
        self.__db_ini = val

    @db_act.setter
    def db_act(self, val):
        self.__db_act = val

    @db_like.setter
    def db_like(self, val):
        self.__db_like = val

    @db_credits.setter
    def db_credits(self, val):
        self.__db_credits = val

    @sample.setter
    def sample(self, val):
        self.__sample = val

    @db_full_data.setter
    def db_full_data(self, val):
        self.__db_full_data = val

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

    def merge_full_data(self):
        data = self.pre.merge_over_like_credits(self.db_ini, self.db_like, self.db_credits)
        self.db_full_data = data

        return None

    def get_4_random_movies(self):
        sample = self.db_act.sample(4)

        # self.db_act = self.db_act.drop(sample.index)
        ids = sample.index.values

        return sample

    def update_sample(self):

        m_with_like = self.sample[self.sample['like'].notna()]

        self.db_like = self.db_like.append(m_with_like[['id', 'like']])
        self.db_act = self.db_act.drop(m_with_like.index)
        self.sample = self.get_4_random_movies()

        print(len(self.db_act))
        print(len(self.db_like))

        return None
