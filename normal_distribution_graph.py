# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalFun(x,mu,sigma):
	pdf=np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
	return pdf

x=np.arange(-15,15,0.001)
y=normalFun(x, 0, 1)
sump=0
alpha=0
for p in y[::-1]:
	sump+=p
	alpha+=1
	if sump>=50:		
		break

print(x[-alpha], ...)
plt.plot(x,y)
plt.show()