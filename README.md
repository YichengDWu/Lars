# LEAST ANGLE REGRESSION and Modificated LEAST ANGLE REGRESSION

This is a python version of the LAR method, which strictly follows the formulas in [LEAST ANGLE REGRESSION](http://statweb.stanford.edu/~tibs/ftp/lars.pdf), especially from page 413 to 417.

# Usage
`python main.py` for oridinary LAR. 

`python main.py -t xx -p` for Lasso Modification of LAR.

You can tune the parameter `t`, i.e. the L1 regularization, to get different results.

To run the example
`python main.py -t 100000 -p`  
 To do test   
`python main.py -t xx -test`  or

`python main.py -test`

# Performance 

| Model | R^2 on train set | R^2 on test set |  MSE on train set | MSE on test set | 
|---------|--------|--------| --------|--------|
| LAR   |   0.809 | 0.809 | 11.546 |11.878|
| LARLasso (t = 100000)| 0.809 | 0.809 | 11.546 |11.878|
| LARLasso (t = 150)| 0.808 | 0.808 | 11.576 |11.914|
| LARLasso (t = 120)| 0.803 | 0.807 | 11.890 |11.888|
