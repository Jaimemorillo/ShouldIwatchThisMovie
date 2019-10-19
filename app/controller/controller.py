from taste.preparation import Preparation
import numpy as np


class DBController:

    def __init__(self, data_path='data/'):
        self.pre = Preparation()
        self.path = data_path
        self.__db_ini = self.load_ini_db()
        self.__db_act = self.db_ini.copy()
        self.__db_like = self.load_like_db()
        self.__sample = self.get_4_random_movies()

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
    def sample(self):
        return self.__sample

    @db_ini.setter
    def db_ini(self, val):
        self.__db_ini = val

    @db_act.setter
    def db_act(self, val):
        self.__db_act = val

    @db_like.setter
    def db_like(self, val):
        self.__db_like = val

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

    def get_4_random_movies(self):
        sample = self.db_act.sample(4)

        # self.db_act = self.db_act.drop(sample.index)
        ids = sample.index.values

        return sample

    def update(self):

        m_with_like = self.sample[self.sample['like'].notna()]

        self.db_like = self.db_like.append(m_with_like[['id', 'like']])
        self.db_act = self.db_act.drop(m_with_like.index)
        self.sample = self.get_4_random_movies()

        print(len(self.db_act))
        print(len(self.db_like))

        return None
