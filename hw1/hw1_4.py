#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMS W4771 Machine Learning
HW 1, Problem 4
Tianyang Yang, Wenqi Wang
October 5, 2018
"""


from scipy.misc import derivative
from math import exp


def grad(f,x,eta=10,K=1e5,epsilon=1e-5,s=1):
    while f(x) < f(x-s*derivative(f,x)):
        s = s/eta
    for k in range(int(K)):
        stepSize = s
        while f(x) < f(x-stepSize*derivative(f,x)):
            stepSize = stepSize/eta
        x = x - stepSize*derivative(f,x)
        if stepSize*derivative(f,x)< epsilon :
            return x
    return x

def f(x):
    return (x-4)**2+2*exp(x)

if __name__ == '__main__':
    result  = grad(f,0)
    print(result)
