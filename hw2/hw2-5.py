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
import os

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


        
def q3_pred(train,k=20,K=10,alpha=0.01):
    n,d = train.shape
    u = np.random.normal(0,0.1,(n,k))
    v = np.random.normal(0,0.1,(d,k))
    for h in range(int(K)):
        for i in range(n):
            for j in range(d):
                er = train[i,j]-u[i].dot(v[j].T)
                u[i] = u[i]+alpha*er*v[j]
                v[j] = v[j]+alpha*er*v[j]
        print("u: "+str(u[0:3]))
        print("v: "+str(v[0:3]))
    return u.dot(v.T)
    


# predict using user-based collaborative filtering model    
def filter_predict(train):
    length = np.sqrt(np.sum(np.square(train),axis = 1))
    unit_train = (train)/(length[:,None]+1e-8)
    sim = unit_train.dot(unit_train.transpose())
    P = train.copy()
    P[P!=0] = 1
    pred = (sim.dot(train)+1e-8)/(np.abs(sim).dot(P)+1e-8)
    return pred

# predict using model in Q3
def predict(train,k=30,K=40,alpha=0.0001):
    n,d = train.shape
    u = np.random.random_sample((n,k))
    v = np.random.random_sample((d,k))
    def ugrad():
        udelta = np.zeros((n,k))
        for i in range(n):
            ri = train[i,][train[i,]!=0]
            vi = v[train[i,]!=0]
            predi = u[i].dot(vi.T)-ri
            udelta[i] =  np.sum(predi[:,None]*vi,0)
        return udelta
    def vgrad():
        vdelta = np.zeros((d,k))
        for j in range(d):
            rj = train.T[j,][train.T[j,]!=0]
            uj = u[train.T[j,]!=0]
            predj = v[j].dot(uj.T)-rj
            vdelta[j] = np.sum(predj[:,None]*uj,0)
        return vdelta
    for i in range(int(K)):
        u = u - alpha*ugrad()
        v = v - 0.01*alpha*vgrad()
    pred = u.dot(v.T)
    pred[pred>5] = 5
    pred[pred<0] = 0
    return pred   
    
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
    for train_size in range(300, 650, 50):
        train_size_list.append(train_size)
        test_size = matrix.shape[0] - train_size
        print("train size: {}, test size: {}".format(train_size, test_size))
        mse1, mse2 = 0, 0
        rep = 10
        for i in range(rep):
            print(" rep {}".format(i))
            train, test = split(matrix, test_size)
            pred1 = filter_predict(train)
            pred2 = predict(train)
            mse1 += compute_mse(pred1, test)
            mse2 += compute_mse(pred2, test)
        mse1 = mse1 / rep
        mse1_list.append(mse1)
        mse2 = mse2 / rep
        mse2_list.append(mse2)
        print("  filtering prediction mse: {}".format(mse1))
        print("  Q3 prediction mse: {}".format(mse2))
    return train_size_list, mse1_list, mse2_list
  
# main
if __name__ == '__main__':
    
    # read in data
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = cwd+'/movie_ratings.csv'
    ratings = pd.read_csv(path, sep=',')
    
    # construct data matrix
    n_users = ratings.userId.unique().shape[0]
    n_movies = ratings.movieId.unique().shape[0]
    movie_dict = ratings.movieId.unique()
    matrix = np.zeros((n_users, n_movies))
    for entry in ratings.itertuples():
        matrix[entry[1]-1, np.where(movie_dict==entry[2])[0][0]] = entry[3]

    # compare performance
    xlist, ylist1, ylist2 = compare(matrix)
    plt.plot(xlist, ylist1, 'r', label='Our model')
    plt.plot(xlist, ylist2, 'b', label='Q3 model')
    plt.title('Model performance comparison')
    plt.xlabel('Size of train set')
    plt.ylabel('Prediction MSE')
    plt.legend(loc='upper right')
    plt.savefig('pic1.png', dpi=1024)
    



    