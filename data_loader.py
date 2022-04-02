import os
import pandas as pd

from scipy.io import arff
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self, data_folder, filename):
        self.data_folder = data_folder
        self.filename = filename

    def _read_data(self):
        data_path = os.path.join(self.data_folder, self.filename)
        data = arff.loadarff(data_path)
        df = pd.DataFrame(data[0])

        return df

    def _preprocess_data(self, df: pd.DataFrame):
        # delete samples without labels
        label_column_name = df.columns[-1]
        df = df.dropna(subset=[label_column_name])

        # split features and labels
        X = df.iloc[:, :-1].values

        Y_data = df.iloc[:, -1].values
        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(Y_data)

        # deal with missing/NaN values in features
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X)
        X = imputer.transform(X)

        return X, y

    def get_data(self, test_ratio=0.2):
        df = self._read_data()

        X, y = self._preprocess_data(df)

        X_train, X_test,  y_train, y_test = train_test_split(
            X, y, test_size=test_ratio
        )

        return X_train, X_test, y_train, y_test
