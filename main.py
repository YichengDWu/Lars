import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.model_selection import train_test_split

from utils import preprocess
from model import Lars

def main():
    import argparse
    parser = argparse.ArgumentParser('LAR')
    parser.add_argument('-t', '--restrain', type=float, default=np.inf,
                        help='Maximum 1 norm of coefs')
    parser.add_argument('-p', '--lasso_path', action='store_true',
                        help='Coefficient Trajectories')
    parser.add_argument('--test', action='store_true',
                        help='Whether to do test')
    
    args = parser.parse_args().__dict__
    
    df = preprocess()
    feats = ['cylinders', 'displacement', 'horsepower', 'weight',
                       'acceleration', 'model year']
    X = df[feats].values
    y = df[['mpg']].values
    if args['test']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    else:
        X_train,y_train = X,y
    lar =Lars(feats,args['restrain'])
    lar.fit(X_train, y_train.reshape(-1,1))
    print("R^2 on train set:", lar.score(X_train,y_train))
    if args['test']:
        print("R^2 on test set:", lar.score(X_test,y_test))
    if args['lasso_path']:
        lar.plot_path()
    
if __name__ == '__main__':
    main()
    plt.show()
