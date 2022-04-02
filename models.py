from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class GridSearchBaseModel:
    '''
    Base class that incorporates the grid search algorithm.

    CANNOT BE INSTANTIATED OR CALLED DIRECTLY!!
    '''

    def __init__(self, grid_search=False):
        self.grid_search = grid_search
        self.model = None

    def build_grid_search_model(self, *args, **kwargs):
        if self.grid_search:
            self.grid_search_model = GridSearchCV(self.model, *args, **kwargs)

    def train(self, X, y):
        if self.grid_search:
            self.grid_search_model.fit(X, y)
            self.model = self.grid_search_model.best_estimator_
        else:
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class DecisionTree(GridSearchBaseModel):

    def __init__(self, grid_search=False, *args, **kwargs):
        super(DecisionTree, self).__init__(grid_search)
        self.model = DecisionTreeClassifier(*args, **kwargs)


class RandomForest(GridSearchBaseModel):

    def __init__(self, grid_search=False, *args, **kwargs):
        super(RandomForest, self).__init__(grid_search)
        self.model = RandomForestClassifier(*args, **kwargs)


class GaussianNaiveBayes(GridSearchBaseModel):

    def __init__(self, grid_search=False, *args, **kwargs):
        super(GaussianNaiveBayes, self).__init__(grid_search)
        self.model = GaussianNB(*args, **kwargs)


class SupportVectorMachine(GridSearchBaseModel):

    def __init__(self, grid_search=False, *args, **kwargs):
        super(SupportVectorMachine, self).__init__(grid_search)
        self.model = SVC(*args, **kwargs)
