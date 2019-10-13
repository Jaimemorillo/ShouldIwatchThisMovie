import numpy as np

class Movies:

    def __init__(self, ids, titles, overviews):
        self.__ids = ids
        self.__titles = titles
        self.__overviews = overviews
        self.__reduced_overviews = [" ".join(s.split()[0:30] + ['...']) for s in overviews]
        self.__predictions = [80 for i in titles]
        self.__tastes = None

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
        index = np.where(self.ids == id)[0][0]
        print(index)
        title = self.titles[index]
        print(title)
        overview = self.overviews[index]
        prediction = self.predictions[index]
        return title, overview, prediction

