# LAR and Modificated LAR

This is an python LAR implement, which strictly follows the formulas in [LAR](http://statweb.stanford.edu/~tibs/ftp/lars.pdf), especially from page 413 to 417.

# Usage
`python main.py` for oridinary LAR. 

`python main.py -t 200 -p` for Lasso Modification of LAR.

You can tune the parameter `t`, i.e. the L1 regularization, to get deffirent results.

# Performance 

| Model | R^2 on train set | R^2 on test set |
| LAR   | 0.8 | 0.7 |
| LARLasso | 0.8 | 0.7 | 

| Model | R^2 on train set | R^2 on test set | 
|---------|--------|--------| 
