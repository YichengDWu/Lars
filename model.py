from utils import normalize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

class Lars:
    """
    This implementation is based on Efron, Hastie, Johnstone, and Tibshirani's
    2004 paper, "Least Angle Regression," especially the equations on pages 413
    and 414.
    """
    def __init__(self,feats, t = np.inf):
        self.feats = feats
        self.t = t
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
            print(f"Starting step {i}...")
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
                #last stage
                gamma = C_hat/A_A
                
            #3.3
            d = w.squeeze()*s 
            #Modification
            if self.t!=np.inf:
                #3.4
                gamma_j = -beta[A]/d
                gamma_p = gamma_j[gamma_j>0]
                if len(gamma_p) > 0:
                    gamma_tilde = min(gamma_p)
                    j = np.where(gamma_p == gamma_tilde)[0][0]
                else:
                    gamma_tilde = np.inf
                if gamma_tilde < gamma:
                    j = np.where(A==True)[0][j]
                    A[j] = False
                    beta[j]=0.0
                    print(f"The {j}th variable was dropped!")
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
                    d = w.squeeze()*s       
            #2.12
            mu += gamma * u
            beta[A] += gamma * d
            self.beta_ma = np.concatenate((self.beta_ma, beta.reshape(1,-1)), axis = 0)
            self.plot_bar(i, c, beta)
            print("Active set:", A*1)
            print("Done!")
            
            if sum(abs(beta))>self.t:
                break
                print("Restrain statisfied!")
        
    def predict(self,X):
        X = normalize(X)
        beta = self.beta_ma[-1,:]
        pred = np.dot(X, beta.reshape(-1,1))
        return pred+self.y_mean
    
    def plot_bar(self, i, c, beta):
        fig, axs = plt.subplots(1,2, figsize = (12,4))
        f1 = sns.barplot(x=self.feats,
                        y=beta.flatten(),palette='rocket',
                    ax = axs[0])
        f1.set_xticklabels(axs[0].get_xticklabels(),rotation = 30, fontsize = 10)
        axs[0].set_ylim([-120, 120])
        plt.ylabel('Beta')      
        
        f2 = sns.barplot(x=self.feats,
                         y=c.flatten(),palette='rocket',
                         ax = axs[1])
        f2.set_xticklabels(axs[1].get_xticklabels(), rotation = 30, fontsize = 10)
        plt.ylabel('Correlation') 
        fig.suptitle(f'Step = {i}', fontsize=10)
        fig.tight_layout()
        plt.show(block = False)
        
    def plot_path(self):
        plt.figure(figsize = (9,6))
        coefs = self.beta_ma
        xx = np.repeat(np.arange(coefs.shape[0]),coefs.shape[1]).reshape(coefs.shape[0],coefs.shape[1])

        plt.plot(xx, coefs)
        ymin, ymax = plt.ylim()
        plt.xlabel('Step')
        plt.ylabel('Coefficients')
        plt.title('LAR Path')
        plt.gca().legend(self.feats)
        plt.show(block = False)
        
    def score(self, X, y_true):
        X = normalize(X)
        y_pred = self.predict(X)
        sse = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
        tse = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
        r2_score = 1 - (sse / tse)
        return r2_score[0]
