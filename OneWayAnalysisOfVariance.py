import numpy as np
import pandas as pd
from scipy import stats

class OneWayAnova:
	"""
	Doc: A class used to calculate One-way Anova with pandas
	"""
	def __init__(self,lists):
		self._dataset=pd.DataFrame(lists)
		
		self._levelSampleNum=self._dataset.count(axis=1)
		self._sampleNum=self._levelSampleNum.sum()
		self._levelNum=self._levelSampleNum.count()
		self._Tj=self._dataset.sum(axis=1)
		self._Ttotal=np.sum(self._Tj)
		self._St=np.sum((self._dataset **2).sum(axis=1))-(self._Ttotal**2)/self._sampleNum
		self._Stemp=[tj ** 2 /nj for tj,nj in zip(self._Tj,self._levelSampleNum)]

		self._Sa=np.sum(self._Stemp)-(self._Ttotal**2)/self._sampleNum
		self._Se=self._St-self._Sa
		print(self._Sa, self._Se,self._St)

		# self.dataset=np.array(lists)
		# self.levelNum=self.dataset.shape[0]
		# self.elementNum=self.dataset.shape[0]*self.dataset.shape[1]
		# self.Tj=self.dataset.sum(axis=1)
		# self.Tsum=self.Tj.sum()
		# self.St=(self.dataset **2).sum()-self.Tsum**2/self.elementNum
		# self.Sa=(self.Tj**2/self.dataset.shape[1]).sum()-self.Tsum**2/self.elementNum
		# self.Se=self.St-self.Sa
		# print(self.Sa, self.Se,self.St)

	def test_hypos(self,alpha_param):
		self._Fratio=(self._Sa/(self._levelNum-1))/(self._Se/(self._sampleNum-self._levelNum))
		print(self._Fratio)
		if self._Fratio > stats.f.ppf(1-alpha_param,self._levelNum-1,self._sampleNum-self._levelNum):
			print("Reject the hypothesis")
		else:
			print("Accept the hypothesis")

	def get_variance_estimate(self):
		return self._Se/(self._sampleNum-self._levelNum)

	def get_confidence_interval(self,confidence_level):
		self._temp=stats.t.ppf(1-(1-confidence_level)/2,self._sampleNum-self._levelNum)
		print(self._temp, ...)
		self.confidence_level=[]
		for i in range(len(self._dataset.mean(axis=1))):
			for j in range(i+1,len(self._dataset.mean(axis=1))):
				self._mean_value=self._dataset.mean(axis=1)[i]-self._dataset.mean(axis=1)[j]
				
				self._temp_sqrt=self._temp*np.sqrt(self._Se/(self._sampleNum-self._levelNum)*(1/self._levelSampleNum[i]+1/self._levelSampleNum[j]))
				print(self._temp_sqrt, ...)
				self.confidence_level.append((self._mean_value-self._temp_sqrt,self._mean_value+self._temp_sqrt))

		return self.confidence_level



if __name__ == '__main__':
	# lists=[[0.236,0.238,0.248,0.245,0.243],[0.257,0.253,0.255,0.254,0.261],[0.258,0.264,0.259,0.267,0.262]]
	# lists=[[19,22,20,18,15],[20,21,33,27,40],[16,15,18,26,17],[18,22,19]]
	lists=[[14,13,9,15,11,13,14,11],[10,12,7,11,8,12,9,10,13,9,10,9],[11,5,9,10,6,8,8,7]]
	anv=OneWayAnova(lists)
	anv.test_hypos(0.05)
	print(anv.get_confidence_interval(0.95))
