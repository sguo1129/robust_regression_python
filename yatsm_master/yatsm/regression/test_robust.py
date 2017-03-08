import numpy as np
from yatsm.regression.robust_fit import RLM

xfile = 'C:\DSS Backup\D_Drive\LSRD\CCDC\robust_regression_python\input1.csv'

data = np.genfromtxt(xfile, dtype=float, delimiter=',')

xarr = data[:,0:4]
yarr = data [:,4]

print len(xarr)
print len(yarr)

print RLM.fit(xarr, yarr)