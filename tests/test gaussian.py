########################################
# Testing IRLS with gaussian settings
########################################
import numpy as np
import os
os.chdir('S:/Python/projects/glm_irls')
from models import glm_gaussian

####################
# helpers to test results
####################
def test_results(BHat, Beta, cutoff = .1):
    T1 = np.all(BHat.shape == Beta.shape)
    T2 = np.sum(np.abs(BHat - Beta)) < cutoff
    return T1 and T2

def make_dataset(N, Beta, link):
    np.random.seed(1)
    
    X = np.concatenate([np.random.uniform(low = 1, high = 2, size = [N, 2]), np.ones([N, 1])], axis = 1)
    eta = np.matmul(X, Beta)
    
    if link == "identity":
        def inv_link(eta):
            return eta
    elif link == "log":
        def inv_link(eta):
            return np.exp(eta)
    elif link == "reciprocal":
        def inv_link(eta):
            return 1 / eta
    else:
        print("invalid link")
        
    mu = inv_link(eta)
    Y = np.random.normal(mu, scale = 1)
        
    return X, Y

####################
# Test identity link
####################
Beta = np.array([.5, 1, 1.5])
X, Y = make_dataset(N = 25000, Beta = Beta, link = "identity")

model = glm_gaussian(link = "identity")
model.fit(X, Y)

test_results(model.coef(), Beta, .1)
np.mean(np.abs(Y -  model.predict(X))) < 1

del Beta, X, Y, model

####################
# Test log link
####################
Beta = np.array([.04, .02, .015])
X, Y = make_dataset(N = 25000, Beta = Beta, link = "log")

model = glm_gaussian(link = "log")
model.fit(X, Y)

test_results(model.coef(), Beta, .1)
del Beta, X, Y, model

####################
# Test reciprocal link
####################
Beta = np.array([.5, 1, 1.5])
X, Y = make_dataset(N = 25000, Beta = Beta, link = "reciprocal")

model = glm_gaussian(link = "reciprocal")
model.fit(X, Y)

test_results(model.coef(), Beta, 1)
del Beta, X, Y, model
