########################################
# Testing IRLS with bernoulli settings
########################################
import numpy as np
import scipy.stats as stats
import os
os.chdir('S:/Python/projects/glm_irls')
from models import glm_bernoulli

####################
# helpers to test results
####################
def test_results(BHat, Beta, cutoff = .1):
    T1 = np.all(BHat.shape == Beta.shape)
    T2 = np.sum(np.abs(BHat - Beta)) < cutoff
    return T1 and T2

def make_dataset(N, Beta, link):
    np.random.seed(1)
    
    X = np.concatenate([np.random.normal(size = [N, 2]), np.ones([N, 1])], axis = 1)
    eta = np.matmul(X, Beta)
    
    if link == "logit":
        def inv_link(eta):
            return np.exp(eta) / (1 + np.exp(eta))
    elif link == "probit":
        def inv_link(eta):
            norm = stats.norm
            return norm.cdf(eta)
    else:
        print("invalid link")
        
    mu = inv_link(eta)
    Y = np.random.binomial(1, mu)
        
    return X, Y

####################
# Test logit link
####################
Beta = np.array([.04, .02, .015])
X, Y = make_dataset(N = 25000, Beta = Beta, link = "logit")

model = glm_bernoulli(link = "logit")
model.fit(X, Y)

test_results(model.coef(), Beta, .1)
del Beta, X, Y, model

####################
# Test probit link
####################
Beta = np.array([.04, .02, .015])
X, Y = make_dataset(N = 25000, Beta = Beta, link = "probit")

model = glm_bernoulli(link = "probit")
model.fit(X, Y)

test_results(model.coef(), Beta, .1)
del Beta, X, Y, model

