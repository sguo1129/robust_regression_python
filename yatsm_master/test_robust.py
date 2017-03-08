import numpy as np
import yatsm.regression.robust_fit as rf

xfile = r'C:\DSS Backup\D_Drive\LSRD\CCDC\robust_regression_python\input1.csv'
data = np.genfromtxt(xfile, dtype=np.float, delimiter=',')
yfile = r'C:\DSS Backup\D_Drive\LSRD\CCDC\robust_regression_python\input2.csv'
data2 = np.genfromtxt(yfile, dtype=np.float, delimiter=',')

xarr = data[:,0:4]
one = np.ones((18,1))
xarr=np.hstack((one, xarr))
yarr = data[:,4]
xarr2 = data2[:,0:4]
xarr2=np.hstack((one, xarr2))
yarr2 = data2[:,4]

m = rf.RLM(M=rf.bisquare, tune=4.685,
           scale_est=rf.mad, scale_constant=0.6745, update_scale=True,
           maxiter=5, tol=1e-8)
m.fit(xarr, yarr)
print(m)
m2 = rf.RLM(M=rf.bisquare, tune=4.685,
           scale_est=rf.mad, scale_constant=0.6745, update_scale=True,
           maxiter=5, tol=1e-14)
m2.fit(xarr2, yarr2)
print(m2)

mask = np.logical_or(np.abs(yarr - m.predict(xarr)) > 4.89*87.0,
        np.abs(yarr2 - m2.predict(xarr2)) > 4.89*195.0)

# print np.abs((yarr - m.predict(xarr))) > 4.89*76.0
# print np.abs((yarr2 - m2.predict(xarr2))) > 4.89*143.0
print mask