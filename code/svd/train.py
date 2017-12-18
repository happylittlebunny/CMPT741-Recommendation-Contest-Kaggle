from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import evaluate, print_perf 
from surprise import accuracy
from surprise import GridSearch

import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\CMPT741\\Project\\data\\train_rating.txt')

cols = ['user_id', 'business_id', 'rating']

df = df[cols]

reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(df, reader)
trainset.split(n_folds=10) 

# http://surprise.readthedocs.io/en/stable/getting_started.html#tune-algorithm-parameters-with-gridsearch
# best params: MEAN RMSE: 1.2525329258
param_grid = {
              'n_factors':[1],
              'n_epochs': [45], 
              'lr_bu':[0.004],
              'lr_bi':[0.008],
              'lr_pu':[0.0015],
              'lr_qi':[0.000025],
              'reg_bu':[0.24],
              'reg_bi':[0.24],
              'reg_pu':[0.055],
              'reg_qi':[0.0085],
              'init_mean':[0],
              'init_std_dev':[0]              
              }

grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])

grid_search.evaluate(trainset)

print(grid_search.best_score['RMSE'])

print(grid_search.best_params['RMSE'])