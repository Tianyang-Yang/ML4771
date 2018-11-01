# -*- coding: utf-8 -*-
"""
COMS W4771 Machine Learning
HW 2, Problem 5
Tianyang Yang (ty2388), Wenqi Wang (ww2505)
November 2, 2018
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# split data into test and train sets
#def split(matrix, test_size):
#    test = np.zeros(matrix.shape)
#    train = matrix.copy()
#    for user in range(n_users-1):
#        test_indices = np.random.choice(matrix[user, :].nonzero()[0], size=test_size, replace=False)
#        train[user, test_indices] = 0
#        test[user, test_indices] = matrix[user, test_indices]
#    return test, train

# split data into test and train sets
def split(matrix, test_size):
    test = np.zeros(matrix.shape)
    train = matrix.copy()
    test_users = np.random.choice(range(matrix.shape[0]), size=test_size, replace=False)
    for user in test_users:
        test_movie = np.random.choice(matrix[user, :].nonzero()[0], size=20, replace=False)
        train[user, test_movie] = 0
        test[user, test_movie] = matrix[user, test_movie]
    return train, test

# predict using user-based collaborative filtering model    
def filter_predict(train):
    dot = train.dot(train.transpose()) + 1e-12
    norm = np.array([np.sqrt(np.diagonal(dot))])
    user_sim = dot/norm/norm.transpose()
    mean_ratings = train.mean(axis=1)
    diff = train - mean_ratings[:, np.newaxis]
    pred = mean_ratings[:, np.newaxis] + user_sim.dot(diff) / np.array([np.abs(user_sim).sum(axis=1)]).transpose()
    return pred

# predict using model in Q3
def predict(train):
    pass    
    
# compute mean squared error between prediction and test
def compute_mse(pred, test):
    pred = pred[test.nonzero()].flatten()
    test = test[test.nonzero()].flatten()
    return mean_squared_error(pred, test)

# compare performance of the two models
def compare(matrix):
#    min_nonzero = matrix.shape[1]
#    for user in range(matrix.shape[0]):
#        cur = len(matrix[user, :].nonzero()[0])
#        min_nonzero = min(cur, min_nonzero)
    train_size_list = []
    mse1_list, mse2_list = [], []
    for train_size in range(50, 650, 50):
        train_size_list.append(train_size)
        test_size = matrix.shape[0] - train_size
        print("train size: {}, test size: {}".format(train_size, test_size))
        mse1, mse2 = 0, 0
        rep = 50
        for i in range(rep):
            train, test = split(matrix, test_size)
            pred1 = filter_predict(train)
            #pred2 = predict(train)
            mse1 += compute_mse(pred1, test)
            #mse2 += compute_mse(pred2, test)
        mse1 = mse1 / rep
        mse1_list.append(mse1)
        #mse2 = mse2 / rep
        #mse2_list.append(mse2)
        print("  filtering prediction mse: {}".format(mse1))
        #print("  Q3 prediction mse: {}".format(mse2))
    return train_size_list, mse1_list
  
# main
if __name__ == '__main__':
    
    # read in data
    path = '/Users/wenqi/Desktop/ML/ML4771/hw2/movie_ratings.csv'
    ratings = pd.read_csv(path, sep=',')
    
    # construct data matrix
    n_users = ratings.userId.unique().shape[0]
    n_movies = ratings.movieId.unique().shape[0]
    movie_dict = ratings.movieId.unique()
    matrix = np.zeros((n_users, n_movies))
    for entry in ratings.itertuples():
        matrix[entry[1]-1, np.where(movie_dict==entry[2])[0][0]] = entry[3]

    # perform prediction
#    test_size = 10
#    test, train = split(matrix, test_size)
#    pred = filter_predict(train)
    xlist, ylist = compare(matrix)
    plt.plot(xlist, ylist, 'r', label='Our model')
    plt.title('Model performance comparison')
    plt.xlabel('Size of train set')
    plt.ylabel('Prediction MSE')
    plt.legend(loc='upper right')
    plt.savefig('pic1.png', dpi=1024)
    



    