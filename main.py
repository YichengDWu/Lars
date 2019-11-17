import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from utils import preprocess
from model import Lars

def main():
    df = preprocess()
    feats = ['cylinders', 'displacement', 'horsepower', 'weight',
                       'acceleration', 'model year']
    X = df[feats].values
    y = df[['mpg']].values
    lar =Lars(feats)
    lar.fit(X, y.reshape(-1,1))
    print("R^2:", lar.score(X,y))
    
if __name__ == '__main__':
    main()
    plt.show()
