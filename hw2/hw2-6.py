# -*- coding: utf-8 -*-
"""
COMS W4771 Machine Learning
HW 2, Problem 5
Tianyang Yang (ty2388), Wenqi Wang (ww2505)
November 2, 2018
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

# split data into test and train sets
def split(matrix, test_size):
    test = np.zeros(matrix.shape)
    train = matrix.copy()
    for user in range(n_users-1):
        test_indices = np.random.choice(matrix[user, :].nonzero()[0], size=test_size, replace=False)
        train[user, test_indices] = 0
        test[user, test_indices] = matrix[user, test_indices]
    return test, train

# predict using user-based collaborative filtering model    
def filter_predict(train):
    user_sim = pairwise_distances(train, metric='cosine')
    mean_ratings = train.mean(axis=1)
    diff = train - mean_ratings[:, np.newaxis]
    pred = mean_ratings[:, np.newaxis] + user_sim.dot(diff) / np.array([np.abs(user_sim).sum(axis=1)]).transpose()
    return pred

# predict using model in Q3
def predict(train):
    
    
# compute mean squared error between prediction and test
def compute_mse(pred, test):
    pred = pred[test.nonzero()].flatten()
    test = test[test.nonzero()].flatten()
    return mean_squared_error(pred, test)
   
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
    test_size = 10
    test, train = split(matrix, test_size)
    pred = filter_predict(train)
    mse = compute_mse(pred, test)



    