import numpy as np
import scipy.stats as stats
from abc import ABCMeta, abstractmethod, ABC

class IRLS(ABC):
    __metaclass__ = ABCMeta
    
    def __init__(self, link):
        self.__B = np.zeros([0])
        self.__link = link
        super().__init__()
            
    def coef(self):
        return self.__B
    
    def fit(self, X, Y):
        self.__B = np.zeros([X.shape[1]])
        self.__B[X.shape[1] - 1] = np.mean(Y)
        
        tol = 1000
        while(tol > 0.00001):
            eta = X.dot(self.__B)
            mu = self.__inv_link(eta)
            
            _w = (1 / (self.__var_mu(mu) * self.__a_of_phi(Y, mu, self.__B) )) * np.power(self.__del_eta_del_mu(mu),2)
            W = np.diag(_w)
            z = (Y - mu) * self.__del_eta_del_mu(mu) + eta
            B_update = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(z)
    
            tol =  np.sum(np.abs(B_update - self.__B))
            # print(tol)
    
            self.__B = B_update.copy()
            
    def __inv_link(self, eta):
        if self.__link == "identity":
            return eta
        
        elif self.__link == "log":
            return np.exp(eta)
        
        elif self.__link == "inverse":
            return 1 / eta
        
        elif self.__link == "logit":
            return np.exp(eta) / (1 + np.exp(eta))
        
        elif self.__link == "probit":
            norm = stats.norm
            return norm.cdf(eta)
        
        elif self.__link == "sqrt":
            return np.power(eta, 2)
        
        elif self.__link == "1/mu^2":
            return 1 / np.power(eta, 1/2)
    
    def __del_eta_del_mu(self, mu):
        if self.__link == "identity":
            return np.ones([mu.shape[0],])
        
        elif self.__link == "log":
            return 1/mu
        
        elif self.__link == "inverse":
            return -1 / np.power(mu,2)
        
        elif self.__link == "logit":
            return 1 / (mu * (1 - mu))
        
        elif self.__link == "probit":
            norm = stats.norm
            return norm.pdf(norm.ppf(mu))
        
        elif self.__link == "sqrt":
            return (1/2) * np.power(mu,-1/2)
        
        elif self.__link == "1/mu^2":
            return -2/np.power(mu, 3)
    
    @abstractmethod
    def __var_mu(self, mu):
        pass
    
    @abstractmethod
    def __a_of_phi(self,Y, mu, B):
        pass
    
    def predict(self, X):
        return self.__inv_link(X.dot(self.__B))
    

class glm_gaussian(IRLS):
            
    def _IRLS__var_mu(self, mu):
        return np.ones([mu.shape[0],])
    
    def _IRLS__a_of_phi(self,Y, mu, B):
        return np.sum(np.power(Y - mu, 2)) / (Y.shape[0] - B.shape[0])
         

class glm_bernoulli(IRLS):
         
    def _IRLS__var_mu(self, mu):
        return mu * (1 - mu)
    
    def _IRLS__a_of_phi(self,Y, mu, B):
        return np.ones([Y.shape[0],])
    
    def predict_proba(self, X):
        props = self._IRLS__inv_link(X.dot(self.coef()))
        props = np.array([1 - props, props]).T
        return props
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.where(probs[:,1] <= .5, 0, 1)

    
class glm_poisson(IRLS):
    
    def _IRLS__var_mu(self, mu):
        return mu
    
    def _IRLS__a_of_phi(self,Y, mu, B):
        return np.ones([Y.shape[0],])

    
class glm_gamma(IRLS):
          
    def _IRLS__var_mu(self, mu):
        return np.power(mu,2)
    
    def _IRLS__a_of_phi(self,Y, mu, B):
        # Method of moments estimate
        # See page 165 and 166 from In All Likelihood book
        numerator2 = np.power(Y - mu, 2)
        denominator2 = np.power(mu, 2) * (Y.shape[0] - B.shape[0])
        phi2 = np.sum(numerator2 / denominator2)
        out = np.ones([Y.shape[0]]) * phi2
        return out

    
class glm_inverse_gaussian(IRLS):
        
    def _IRLS__var_mu(self, mu):
        return np.power(mu,3)
    
    def _IRLS__a_of_phi(self,Y, mu, B):
        return -1 * np.sum(np.power(Y - mu, 2)) / (Y.shape[0] - B.shape[0])
