import pandas as pd
import joblib
import itertools


class Utils:
    def chunks(self, iterable):
        batch_size = 100
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def load_from_csv(self, path):
        return pd.read_csv(path)

    def load_from_mysql(self):
        pass

    def features_def(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        Y = dataset[y]
        return X, Y

    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf, "./models/best_model.pkl")
