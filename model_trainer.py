from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer:

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        start = datetime.now()

        self.model.train(self.X_train, self.y_train)

        end = datetime.now()

        self.train_time = end - start

    def get_statistics(self):
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, train_pred)
        test_accuracy = accuracy_score(self.y_test, test_pred)
        train_f1 = f1_score(self.y_train, train_pred)
        test_f1 = f1_score(self.y_test, test_pred)

        stats = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_time': self.train_time
        }

        return stats
