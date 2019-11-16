import pandas as pd

def preprocess():
    print("Starting processing data...")
    col_names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
    df = pd.read_table("../input/auto-mpg.data", header=None, delim_whitespace=True)
    df.columns = col_names
    df = df.drop('origin',axis = 1)
    df = df[df['horsepower'] != '?']
    df['horsepower'] = df['horsepower'].astype('float')
    print("Done!")
    return df
