import numpy as np


class Movies:

    def __init__(self, ids, titles, overviews):
        self.__ids = ids
        self.__titles = dict(zip(ids, titles))
        self.__overviews = dict(zip(ids, overviews))
        self.__reduced_overviews = dict(zip(ids, [" ".join(s.split()[0:30] + ['...']) for s in overviews]))
        self.__predictions = dict(zip(ids, [80 for i in titles]))
        self.__tastes = dict(zip(ids, [None for i in titles]))

    @property
    def ids(self):
        return self.__ids

    @property
    def titles(self):
        return self.__titles

    @property
    def overviews(self):
        return self.__overviews

    @property
    def reduced_overviews(self):
        return self.__reduced_overviews

    @property
    def predictions(self):
        return self.__predictions

    @property
    def tastes(self):
        return self.__tastes

    @ids.setter
    def ids(self, val):
        self.__ids = val

    @titles.setter
    def titles(self, val):
        self.__titles = val

    @overviews.setter
    def overviews(self, val):
        self.__overviews = val

    @reduced_overviews.setter
    def reduced_overviews(self, val):
        self.__reduced_overviews = val

    @predictions.setter
    def predictions(self, val):
        self.__predictions = val

    def get_by_id(self, id):
        title = self.titles.get(id)
        overview = self.overviews.get(id)
        prediction = self.predictions.get(id)
        return title, overview, prediction
