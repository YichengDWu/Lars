import pandas as pd
import numpy as np

def preprocess():
    print("Starting processing data...")
    col_names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
    df = pd.read_table("./auto-mpg.data", header=None, delim_whitespace=True)
    df.columns = col_names
    df = df.drop('origin',axis = 1)
    df = df[df['horsepower'] != '?']
    df['horsepower'] = df['horsepower'].astype('float')
    print("Done!")
    return df

def normalize(X):
    norms = np.linalg.norm(X, axis = 0)
    means = np.mean(X, axis = 0)
    return (X-means)/norms
