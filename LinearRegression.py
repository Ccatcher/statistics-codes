import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class LinearRegression:

	def __init__(self,lists):
		self._x, self._y=np.array(lists)
		self._Sxx=np.sum((self._x - self._x.mean()) ** 2)
		self._Syy=np.sum((self._y - self._y.mean()) ** 2)
		self._Sxy=np.sum((self._x - self._x.mean()) * (self._y - self._y.mean()))

		self._bhat=self._Sxy / self._Sxx
		self._ahat=self._y.mean() - self._bhat * self._x.mean()

		self._Qe = self._Syy - self._bhat * self._Sxy

		self._deltaHatSquare = self._Qe / (len(self._x) - 2)

		self._t = np. sqrt(self._bhat ** 2 * self._Sxx / self._deltaHatSquare) 


	def draw_scatter_plot(self):
		plt.scatter(self._x,self._y,c='k',marker='o')
		self._temp_y=[self._ahat+self._bhat * x for x in self._x]
		plt.plot(self._x,self._temp_y)
		plt.show()

	def get_linearregression_equation(self):
		return self._ahat, self._bhat

	def get_variance_estimate(self):
		return self._deltaHatSquare

	def test_linear_hypothesis(self,alpha):
		
		print("t Test: %f" %self._t)
		if self._t > stats.t.ppf(1-(alpha/2), len(self._x) - 2):
			print("Linear relation between x and y")
			return False
		else:
			print("No linear relation between x and y!")
			return True

	def get_confidence_interval_b(self,confidence_level):
		self._temp1=stats.t.ppf(1-(1-confidence_level) / 2, len(self._x) - 2) * np.sqrt(self._deltaHatSquare / self._Sxx)
		return self._bhat - self._temp1, self._bhat + self._temp1

	def get_confidenct_interval_mu(self, x0, confidence_level):
		self._temp2=stats.t.ppf(1-(1- confidence_level) /2, len(self._x) - 2) * np.sqrt(self._deltaHatSquare * (1/len(self._x) + (x0- self._x.mean())**2 / self._Sxx))
		return self._ahat + self._bhat * x0 - self._temp2, self._ahat + self._bhat * x0 + self._temp2

	def get_prediction_interval_y(self, x0, confidence_level):
		self._temp3 = stats.t.ppf(1-(1- confidence_level) /2, len(self._x) - 2) * np.sqrt(self._deltaHatSquare * (1 + 1/len(self._x) + (x0- self._x.mean())**2 / self._Sxx))
		# return self._ahat + self._bhat * x0 - self._temp3, self._ahat + self._bhat * x0 + self._temp3
		return self._ahat + self._bhat * x0 - self._temp3, self._ahat + self._bhat * x0 + self._temp3


if __name__ == '__main__':
	# lists=[list(range(100,191,10)),[45,51,54,61,66,70,74,78,85,89]]
	lists=[[9,8.5,9.25,9.75,9,10,9.5,9,9.25,9.5,9.25,10,10,9.75,9.5],[6.5,6.25,7.25,7.0,6.75,7,6.5,7,7,7,7,7.5,7.25,7.25,7.25]]
	linReg=LinearRegression(lists)
	print(linReg.get_linearregression_equation())
	print(linReg.get_variance_estimate(), ...)
	print(linReg.test_linear_hypothesis(0.05))
	print(linReg.get_confidence_interval_b(0.95), ...)
	print(linReg.get_confidenct_interval_mu(0.5, 0.95), ...)
	# print([linReg.get_prediction_interval_y(x,0.95) for x in range(125,146,5)], ...)
	linReg.draw_scatter_plot()
	print(linReg.get_prediction_interval_y(0.5,0.95), ...)




