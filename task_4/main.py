from datetime import datetime

import numpy as np
import pandas as pd
import dill

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import os

import warnings
warnings.filterwarnings('ignore')


class Model():
    def __init__(self, path_to_csv: str = 'path/to/csv'):
        """
        __init__ is a special method in Python classes, it is the constructor method for a class.

        :param  path_to_csv: string, path with prepared dataset

        :return
        """

        self.path = path_to_csv

    def main(self):
        """
        This function reads a file gotten in __init__, splits up the data, trains model,
        write a model into file

        :return
        """

        mapes = []
        df = pd.read_csv(self.path)
        X = df.drop(['estimated_stock_pct', 'std_estimated_stock_pct', 'temperature'], axis=1)
        y = df.estimated_stock_pct

        for i in range(10):
            model = RandomForestRegressor()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, shuffle=True)
            trained_model = model.fit(X_train, y_train)

            y_pred = trained_model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            print(f'Fold: {i + 1}, MAPE: {mape:.3f}')

            mapes.append(mape)

        print(f'Average of MAPE: {sum(mapes) / len(mapes):.3f}')
        print('Saving the model into file...')
        os.mkdir(f'{os.getcwd()}/model/')
        file_name = f'{os.getcwd()}/model/fr_stock_predictor.pkl'
        with open(file_name, 'wb') as file:
            dill.dump({
                'model': model,
                'metadata': {
                    'name_of_author': 'Leonid Timofeev',
                    'name_model': type(model).__name__,
                    'mape': np.round(sum(mapes) / len(mapes), 3),
                    'time_of_creation': datetime.now()
                }
            }, file)
        if os.path.exists(f'{os.getcwd()}/model/fr_stock_predictor.pkl'):
            print('Pickle file is successfully loaded')

        else:
            print('Some error arised')
            os.remove(f'{os.getcwd()}/model/')


if __name__ == '__main__':
    path = os.getcwd()
    model = Model(f'{path}/../task_3/data/prepared_dataset.csv')
    model.main()