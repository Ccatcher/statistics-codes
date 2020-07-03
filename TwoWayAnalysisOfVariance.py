import pandas as pd
import numpy as np
from scipy import stats

class TwoWayAnovaOfUnrepeatedTest:

	def __init__(self,lists):
		self._dataset=np.array(lists)
		self._levelNumA,self._levelNumB=self._dataset.shape
		self._Ttotal=self._dataset.sum()
		self._Ta=self._dataset.sum(axis=1)
		self._Tb=self._dataset.sum(axis=0)

		self._temp=self._Ttotal**2/(self._levelNumA * self._levelNumB)
		self._St=np.sum(self._dataset**2)-self._temp
		self._Sa=np.sum(self._Ta  ** 2) / self._levelNumB - self._temp
		self._Sb=np.sum(self._Tb ** 2) / self._levelNumA - self._temp
		self._Se=self._St- self._Sa -self._Sb

		self._dof=(self._levelNumA - 1) * (self._levelNumB - 1)    # degrees of freedom

	def test_hepothesisA(self, alpha_param):
		self._FaRatio=(self._Sa / (self._levelNumA-1)) / (self._Se / self._dof)
		
		if self._FaRatio >= stats.f.ppf(1-alpha_param, self._levelNumA -1 , self._dof):
			print("Factor A has significant impact!")
		else:
			print("Factor A has no significant impact!")

		return self._FaRatio
	
	def test_hepothesisB(self, alpha_param):
		self._FbRatio=(self._Sb / (self._levelNumB-1)) / (self._Se / self._dof)
		
		if self._FbRatio >= stats.f.ppf(1-alpha_param, self._levelNumB -1 , self._dof):
			print("Factor B has significant impact!")
		else:
			print("Factor B has no significant impact!")

		return self._FbRatio


if __name__ == '__main__':
	lists=[[1.63,1.35,1.27],[1.34,1.3,1.22],[1.19,1.14,1.27],[1.3,1.09,1.32]]
	anv=TwoWayAnovaOfUnrepeatedTest(lists)
	print(anv.test_hepothesisA(0.05))
	print(anv.test_hepothesisB(0.05))