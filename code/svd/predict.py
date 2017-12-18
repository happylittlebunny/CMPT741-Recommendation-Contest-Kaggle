from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import evaluate, print_perf
from surprise import accuracy
from surprise import dump

import pandas as pd
import numpy as np

df_o = pd.read_csv('C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\CMPT741\\Project\\data\\train_rating.txt')
df_test = pd.read_csv('C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\CMPT741\\Project\\data\\test_rating.txt')
cols = ['user_id', 'business_id', 'rating']
cols_test = ['user_id', 'business_id']
df = df_o[cols]
df_test = df_test[cols_test]
df_test['dummy1']=df_test.apply(lambda _: 1,axis=1)

reader = Reader(rating_scale=(1,5))
trainset = Dataset.load_from_df(df,reader).build_full_trainset()
testset = [tuple(x) for x in df_test.values]


algo = SVD(n_factors=1, n_epochs=45, lr_bu=0.004, lr_bi=0.008,
           lr_pu=0.002, lr_qi=0.00003, reg_bu=0.24, reg_bi=0.24, 
           reg_pu=0.055, reg_qi=0.0087, init_mean=0, init_std_dev=0)

algo.train(trainset)
dump.dump('C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\CMPT741\\Project\\model\\svd_dump_1209', algo=algo)

predictions = algo.test(testset)

"""
Quick save
print ('save prediction')
ratings = [x for _,_,_,x,_ in predictions]
df_train_p = pd.DataFrame(ratings, columns=(['rating']))

df_train_p.to_csv('C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\CMPT741\\Project\\result\\svd_4.csv', index=False)

"""

# save detail
output_cols = ['test_id', 'user_id', 'business_id', 'rating']
df_result = pd.DataFrame(columns=output_cols)
index = 0
for uid, iid, true_r, est, _ in predictions:
    print (index, uid, iid, est)
    df_result.loc[index] = [index, uid, iid, est]
    index = index+1

df_result.test_id = df_result.test_id.astype(int)
df_result.user_id = df_result.user_id.astype(int)
df_result.business_id = df_result.business_id.astype(int)
df_result.to_csv('C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\CMPT741\\Project\\result\\result_svd_4.csv',sep=',', header=output_cols, index=False)

submit_output_cols=['test_id', 'rating']
df_submit = df_result[submit_output_cols]
df_submit.to_csv('C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\CMPT741\\Project\\result\\svd_4.csv',sep=',', header=submit_output_cols, index=False)
