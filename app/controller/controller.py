from taste.preparation import Preparation
from movies.movies import Movies


class Controller:

    def __init__(self, data_path='data/tmdb_spanish_overview.csv'):
        self.pre = Preparation()
        self.path = data_path
        self.db_ini = self.load_data_base()
        self.db_act = self.db_ini.copy()

    def load_data_base(self):
        movies_ini = self.pre.get_overview(self.path)
        movies_ini.set_index('id', inplace=True)
        movies_ini.dropna(subset=['title', 'overview'], inplace=True)
        return movies_ini

    def get_4_random_movies(self):
        sample = self.db_act.sample(4)
        self.db_act = self.db_act.drop(sample.index)
        random_movies = Movies(sample.index.values, sample['title'].values, sample['overview'].values)

        return random_movies
