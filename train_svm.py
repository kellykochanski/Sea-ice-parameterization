#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

#/lucas26-scratch/from-science/home/lucas26/lucas26/projects/presentations/agufm2013/poster/

import sys
import pylab
import numpy as np
import IPython

import sklearn.svm as skl_svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


##################################################################
def read_tab(tin):
    fopen = open(tin)
    lines = fopen.readlines()
    header = lines[0]
    header = header.strip().replace('#','')
    header = [h for h in header.split(' ') if h not in '']
    data = []
    for line in lines:
        if line[0] == '#' : continue
        line = line.strip()
        sline = [float(l) for l in line.split(' ') if l not in '']
        data += [sline]
    data = np.array(data)
    return header, data
##################################################################


##################################################################
header, cice = read_tab("./cice.tab")
header_dict = dict(zip(header, np.arange(len(header))))
##################################################################

#IPython.embed()
#sys.exit()

##################################################################
# +++ IMPORTANT UPDATE: Added hemisphere/quantity/month as SVM inputs/features!!!
hem = ['NH', 'SH']
hem_dict = dict(zip(hem, range(len(hem))))

qoi = ['EXTE', 'AREA']
qoi_dict = dict(zip(qoi, range(len(qoi))))

mon = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
mon_dict = dict(zip(mon, np.arange(0,1.+1./11,1./11)))  # scaled between 0 and 1 (that's what SVM likes)

responses = [ ['%s%s%s' % (h,q,m), hem_dict[h], qoi_dict[q], mon_dict[m]] for h in hem for q in qoi for m in mon ]

npars = 7

mod_data = []
for row in cice:
    for r in responses:
        mdat = np.r_[row[1:npars+1],r[1], r[2], r[3], row[header_dict[r[0]]]]
        mdat = np.array([mdat])
        if mod_data == []:
            mod_data = mdat
        else:
            mod_data = np.r_[mod_data, mdat]
##################################################################



##################################################################
# SVM
##svm_opts = '-s 4 -t 2 -g 10.0 -c 50 -n 0.2'   # SVM-R fitting options
#svm_opts = '-s 4 -t 2 -g 5.0 -c 50 -n 0.6'   # SVM-R fitting options
##################################################################
def fit_svm(xin, yin):

    model = skl_svm.NuSVR(kernel='rbf',
                          gamma=5.0,
                          nu=0.6,
                          C=50.0,
                          tol=0.08,
                          verbose=True)

    model.fit(xin, yin)

    return model
##################################################################

#IPython.embed()
#sys.exit()

##################################################################
x = mod_data[:,:-1]
y = mod_data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
model = fit_svm(X_train, y_train)
##################################################################


##################################################################
cross_val = True
if cross_val:
  scores = cross_val_score(model, X_train, y_train, cv=5)
  print(scores)
##################################################################


##################################################################
save_svm = False
if save_svm:
  joblib.dump(model, 'cice.svm')
  model = joblib.load('cice.svm')
##################################################################



##################################################################
# Validation Plot
y_train_pred = model.predict(X_train)
y_test_pred =  model.predict(X_test)
score = model.score(X_test, y_test)

minv = min(min(y_train), min(y_train_pred))
maxv = max(max(y_train), max(y_train_pred))
pylab.clf()
pylab.plot([minv, maxv], [minv, maxv], color='black')
pylab.scatter(y_train_pred, y_train, s=10, edgecolors='blue', facecolors='lightblue', alpha=0.6, label='train')
pylab.scatter(y_test_pred, y_test, s=10, edgecolors='red', facecolors='salmon', alpha=0.6, label='test')
pylab.text(0.1, 0.95, r'R$^{2}$=%s' % str(round(score, 3)), ha='left', va='center', transform = pylab.gca().transAxes)
pylab.xlabel('predicted')
pylab.ylabel('actual')
pylab.title('Independent Validation (N=%s)' % len(y_test))
pylab.legend(loc='lower right', scatterpoints=1)
#pylab.show()
pylab.savefig('svm-validation-plot.png', dpi=240)
##################################################################
