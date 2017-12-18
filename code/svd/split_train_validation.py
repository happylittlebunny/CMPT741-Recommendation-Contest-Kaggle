
# coding: utf-8

# In[31]:


from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import evaluate, print_perf 
from surprise import accuracy
from surprise import GridSearch
from surprise import SVDpp

import pandas as pd
import numpy as np

data = pd.read_csv('train_rating.txt')

cols = ['user_id', 'business_id', 'rating', 'date']


# In[32]:


df = data[cols]


# In[33]:


# add month to each record
df['date'] = pd.to_datetime(df['date'])
df['MOY'] = df['date'].dt.month


# In[34]:


# The month in test_training are 1-7 
# validation_df = (df.loc[(df['MOY'] >= 1) & (df['MOY'] <= 7)]) \
#     .sample(frac = 0.5, random_state = np.random.seed(5)) \
#     .drop(['MOY', 'date'], 1)
validation_df = df.sample(frac = 0.2, random_state = np.random.seed(5)).drop(['MOY', 'date'], 1)


# In[35]:


train_df = df.loc[~df.index.isin(validation_df.index)]             .drop(['MOY', 'date'], 1)


# In[36]:


reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(train_df, reader).build_full_trainset()
validationset = [tuple(x) for x in validation_df.values]


# In[37]:


n_factors = [1]
n_epochs = [45]
lr_bu = [0.004]
lr_pu = [0.002]
lr_qi = [0.00003]
lr_bi = [0.008]
reg_bu = [0.24]
reg_bi = [0.24]
reg_pu = [0.055]
reg_qi = [0.0085]
init_mean = [0]
init_std_dev = [0]
combinations = np.array(np.meshgrid(n_factors, n_epochs, lr_bu, lr_pu, lr_qi, lr_bi, reg_bu, reg_bi, reg_pu, reg_qi,
                                          init_mean, init_std_dev)).T.reshape(-1, 12)
print("number of combinations: " + str(len(combinations)))


# In[30]:


get_ipython().run_cell_magic('time', '', 'rmses = []\ncount = 1\nfor item in combinations:\n    print(count)\n    count += 1\n    algo = SVD(n_factors = int(item[0]), n_epochs = int(item[1]), lr_bu = float(item[2]), lr_pu = float(item[3]), lr_qi = float(item[4]), lr_bi = float(item[5]), reg_bu = float(item[6]), reg_pu = float(item[7]), reg_qi = float(item[8]), reg_bi = float(item[9]), init_std_dev = float(item[10]))\n    algo.train(trainset)\n    predictions = algo.test(validationset)\n    \n    rmse = accuracy.rmse(predictions)\n    rmses.append(rmse)')


# In[11]:


print("\n************************* The best result *************************")
print(min(rmses))
print(combinations[np.argmin(rmses)])

