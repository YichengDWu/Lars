class Lars:
    """
    This implementation is based on Efron, Hastie, Johnstone, and Tibshirani's
    2004 paper, "Least Angle Regression," especially the equations on pages 413
    and 414.
    """
    def __init__(self):
        pass
    def fit(self, X, y):
        X = normalize(X)
        self.y_mean = np.mean(y)
        y = y-self.y_mean
        n = len(y)
        m = X.shape[1]
        beta = np.zeros((m,)).astype('float')
        #mu = X\beta
        mu = np.zeros_like(y)
    
        self.beta_ma = np.zeros((1, m)).astype('float')
        self.cor_ma = np.zeros((1,m)).astype('float')
        for i in range(m):
            #2.8
            c = np.dot(X.T, y - mu)
            self.cor_ma = np.concatenate((self.cor_ma, c.reshape(1,-1)), axis = 0)
            #2.9
            C_hat = abs(c).max()
            A = np.isclose(abs(c), C_hat).squeeze()
            #2.10
            s = np.array([1 if c_>0 else -1 for c_ in c[A]])
            #2.4
            X_A =  s*X[:,A]
            #2.5
            G = np.dot(X_A.T,X_A)
            G_inverse = np.linalg.inv(G)
            A_A = np.power(G_inverse.sum(), -0.5)
            #2.6
            w = np.sum(A_A * G_inverse, axis = 1, keepdims = True)
            u = np.dot(X_A, w)
            #2.11
            a = np.dot(X.T, u)
            #2.13
            A_c = (A == False)
            c_, a_ = c[A_c].T, a[A_c].T
            ne = (C_hat-c_)/(A_A-a_)
            po = (C_hat+c_)/(A_A+a_)
            candicates = np.append(ne[ne>0], po[po>0])
            if len(candicates)!=0:
                gamma = min(np.append(ne[ne>0], po[po>0]))
            else:
                print("Convergence!")
                break 
            #2.12
            mu += gamma * u
            #3.3
            d = w.squeeze()*s
            beta[A] += gamma * d
            self.beta_ma = np.concatenate((self.beta_ma, beta.reshape(1,-1)), axis = 0)
            print((((y-mu)**2)).sum()/392)
            xb = np.dot(X,beta.reshape(-1,1))
            print(np.mean((xb-mu)**2))
            print(beta)
    def predict(self,X):
        X = normalize(X)
        beta = self.beta_ma[-1,:]
        pred = np.dot(X, beta.reshape(-1,1))
        return pred+self.y_mean