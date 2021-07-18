########################################
# Testing IRLS with gamma settings
########################################
import numpy as np
from numpy.testing import assert_equal
from models import glm_gamma
import pytest

####################
# helpers to test results
####################
def check_results(model, Beta, X, Y, cutoff = .1):
    T1 = np.all(model.coef().shape == Beta.shape)
    T2 = np.sum(np.abs(model.coef() - Beta)) < cutoff
    T3  = np.sum(np.power(Y -  model.predict(X),2)) < np.sum(np.power(Y -  np.mean(Y),2))
    return T1 and T2 and T3

def make_dataset(N, Beta, link):
    np.random.seed(1)
    
    X = np.concatenate([np.random.uniform(low = 1, high = 2, size = [N, 2]), np.ones([N, 1])], axis = 1)
    eta = np.matmul(X, Beta)
    
    if link == "inverse":
        def inv_link(eta):
            return 1 / eta
    elif link == "identity":
        def inv_link(eta):
            return eta
    elif link == "log":
        def inv_link(eta):
            return np.exp(eta)
    else:
        print("invalid link")
        
    mu = inv_link(eta)
    Y = np.random.gamma(mu, scale = 1)
        
    return X, Y

def test_fit():
    
    links = ["inverse", "identity", "log"]
    for link in links:
        if link == "inverse":
            Beta = np.array([.5, 1, 1.5])
            X, Y = make_dataset(N = 10000, Beta = Beta, link = link)
            cutoff = 1
        elif link == "identity":
            Beta = np.array([.5, 1, 1.5])
            X, Y = make_dataset(N = 10000, Beta = Beta, link = link)
            cutoff = 1
        elif link == "log":
            Beta = np.array([.5, 1, 1.5])
            X, Y = make_dataset(N = 10000, Beta = Beta, link = link)
            cutoff = 1

        model = glm_gamma(link)
        model.fit(X, Y)
    
        assert_equal(check_results(model, Beta, X, Y, cutoff), True)

def test_valid_links():
    with pytest.raises(ValueError):
        glm_gamma("foo")