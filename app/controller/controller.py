from taste.preparation import Preparation
import numpy as np


class DBController:

    def __init__(self, data_path='data/tmdb_spanish_overview.csv'):
        self.pre = Preparation()
        self.path = data_path
        self.db_ini = self.load_data_base()
        self.db_act = self.db_ini.copy()

    def load_data_base(self):
        movies_ini = self.pre.get_overview(self.path)

        movies_ini['reduced_overview'] = movies_ini['overview'].apply(
            lambda x: " ".join(x.split()[0:30] + ['...']))
        movies_ini['prediction'] = 80
        movies_ini['taste'] = np.nan

        movies_ini = movies_ini[['id', 'title', 'overview',
                                 'reduced_overview', 'prediction', 'taste']]
        return movies_ini

    def get_4_random_movies(self):
        sample = self.db_act.sample(4)
        self.db_act = self.db_act.drop(sample.index)
        # random_movies = Movies(sample.index.values, sample['title'].values, sample['overview'].values)
        ids = sample.index.values

        return ids, sample
