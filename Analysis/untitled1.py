# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:08:59 2025

@author: liad0
"""

varconOld = np.load(r'D:\fitting_test_Contrast\Normal\Tara\2023-12-07\gratingContrastTuning.expVar.noSplit.npy')
varconNew = np.load(r'D:\fitting_test_Contrast\Modified\Tara\2023-12-07\gratingContrastTuning.expVar.noSplit.npy')


cond = varconOld>0
f, ax = plt.subplots(1)
ax.scatter(varconOld,varconNew,alpha=0.7)
ax.plot(np.arange(-0.3,1,0.1),np.arange(-0.3,1,0.1),'k--',lw=3)
ax.vlines(0,-0.3,1,color='k',ls='--',lw=1)
ax.hlines(0,-0.3,1,color='k',ls='--',lw=1)
ax.set_xlabel('old ev')
ax.set_ylabel('new ev')
