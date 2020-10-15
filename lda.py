import numpy as np
from itertools import combinations

class LDA:
    def fit(self, x, y):
        # x -> shape=[batch_size, n_features]
        # y -> shape=[batch_size, 1]
        self.n_features = x.shape[1]
        self.n_classes = len(np.unique(y))
        self.phi = self.cal_phi(y)
        self.mean = self.cal_mean(x, y)
        self.sigma = self.cal_sigma(x, y)
        self.cal_intercept_coef()
    
    def cal_phi(self, y):
        phi = [] # each phi correspon to each class
        for i in range(self.n_classes):
            phi.append((y==i).mean())
        return np.array(phi)
    
    def cal_mean(self, x, y):
        mean = [] # first dimension correspond to class
        for i in range(self.n_classes):
            tmp_mean = x[(y==i).squeeze()].mean(0)
            mean.append(tmp_mean)
        return np.array(mean)
    
    def cal_sigma(self, x, y):
        y = y.squeeze()
        n = x.shape[1]
        sigma = np.zeros((n, n))
        for i in range(self.n_classes):
            tmp = x[y==i] - self.mean[i]
            sigma += tmp.T.dot(tmp) # for shared sigma
        sigma /= len(x) # for shared sigma
        return sigma
    
    def cal_intercept_coef(self):
        self.coef = self.mean @ np.linalg.pinv(self.sigma)
        self.intercept = -0.5 * (self.mean.dot(np.linalg.pinv(self.sigma)) * self.mean).sum(1) + np.log(self.phi)
    
    def decision_function(self, x):
        sigma_inv = np.linalg.pinv(self.sigma)
        decisions = []
        for comb in combinations(range(self.n_classes), 2):
            i,j = comb
            decision = np.log(self.phi[i] / self.phi[j]) + (-0.5*((self.mean[i] + self.mean[j]) @ sigma_inv @ (self.mean[i] - self.mean[j])) + x @ sigma_inv @ (self.mean[i] - self.mean[j]))
            decisions.append(decision)
        return np.array(decisions).T
    
    def generate(self, class_, n_samples=1):
        return np.random.multivariate_normal(self.mean[class_], self.sigma, n_samples)
        
    def predict(self, x):
        # x -> shape=[batch_size, n_features]
        probs = []
        n = x.shape[1]
        sigma_inv = np.linalg.pinv(self.sigma)
        for i in range(self.n_classes):
            prob = self.mean[i].dot(sigma_inv).dot(x.T) - self.mean[i].dot(sigma_inv).dot(self.mean[i]) * 0.5 + np.log(self.phi[i])
            probs.append(prob)
        return np.array(probs).T
    
    def empirical_rule(self, x):
        # x -> shape=[batch_size, n_features]
        v, _ = np.linalg.eig(self.sigma)
        result = []
        for i in range(self.n_classes):
            upper_limit = self.mean[i] + (v * 3)
            lower_limit = self.mean[i] - (v * 3)
            result.append(((lower_limit > x) | (upper_limit < x)).any())
            
        return np.array(result).T