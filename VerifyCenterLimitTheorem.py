import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

# n,p=20000000,0.3

# # bernoulli_Rv=stats.bernoulli(p)
# x=np.arange(stats.binom.ppf(0.01,n,p),stats.binom.ppf(0.99,n,p))
# # print(x, ...)
# binom_pmf=stats.binom.pmf(x,n,p)
# plt.plot(x,binom_pmf)

# binom_values=stats.binom.rvs(n,p,size=1000000)
# normalized_binom_values=(binom_values-n*p)/np.sqrt(n*p*(1-p))
# hist,bin_edges=np.histogram(normalized_binom_values,100,density=True)
# print(bin_edges, ...)
# plt.plot(bin_edges[1:],hist)

# plt.show()

def drawNormalDistPdf(subplotAxes):
	x=np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99),1000)
	pdf=stats.norm.pdf(x)
	subplotAxes.plot(x,pdf,'k--',alpha=0.4)

def normalizedBinomDistVsNormalDist(subplotAxes):
	nps=[[100,0.3],[1000,0.3],[10000,0.9]]
	handles=[]
	labels=[]
	size=100000
	bins_count=25
	print("Binomal Dist Vs Normal Dist")
	print("=="*20)

	for n_p in nps:
		binom_values=stats.binom.rvs(n_p[0],n_p[1],size=size)
		normalized_binom_values=(binom_values-n_p[0]*n_p[1])/np.sqrt(n_p[0]*n_p[1]*(1-n_p[1]))
		hist,bin_edges=np.histogram(normalized_binom_values,bins_count,density=True)
		
		# print(stats.describe(normalized_binom_values)) #calculate some statistics of the normalized sample
		print(stats.ttest_1samp(normalized_binom_values,0)) #two-sided test for the null hypothesis that the expected value (mean) of the sample is equal to 0, but this test 
		print(stats.kstest(normalized_binom_values, 'norm')) #Test the binomal dist's large normalized sample with respect to Normal dist
		# print(stats.skewtest(normalized_binom_values))
		# print(stats.kurtosistest(normalized_binom_values))
		print("-------------------------")

		handle,=subplotAxes.plot(bin_edges[1:],hist)
		handles.append(handle)
		labels.append('n=%d, p=%.2f' %(n_p[0],n_p[1]))
	subplotAxes.legend(handles=handles,labels=labels,frameon=False,loc="upper right",labelspacing=0)
	drawNormalDistPdf(subplotAxes)
	subplotAxes.set_title('Binomal Dists VS Normal Dist')

def normalizedPossionDistVsNormalDist(subplotAxes):
	n_lambdas=[[100,0.6],[1000,.6],[1000,6]]
	reCounts=10000
	handles=[]
	labels=[]
	bins=25
	print("Poisson Dist Vs Normal Dist")
	print("=="*20)

	for n,lamb in n_lambdas:		
		samples=[]
		for i in range(0,reCounts):
			rv=stats.poisson.rvs(lamb,size=n)
			samples.append(np.sum(rv))
		normalized_poisson_samples=[(sample-n*lamb)/np.sqrt(n*lamb) for sample in samples]
		hist,bin_edges=np.histogram(normalized_poisson_samples,bins,density=True)
		
		handle,=subplotAxes.plot(bin_edges[1:],hist)
		handles.append(handle)
		labels.append(r'$\lambda$=%.2f,n=%d' %(lamb,n))

		print("lambda=%.2f, n=%d" %(lamb,n))
		print(stats.describe(normalized_poisson_samples))
		print(stats.ttest_1samp(normalized_poisson_samples,0))
		print(stats.kstest(normalized_poisson_samples,'norm'))
		print("-------------------------")

	subplotAxes.legend(handles=handles,labels=labels,loc='best',frameon=False,labelspacing=0)
	drawNormalDistPdf(subplotAxes)
	subplotAxes.set_title('Poisson Dist Vs Normal Dist')

def normalizedUniformDistVsNormalDist(subplotAxes):
	params=[[-5,5,100],[-5,5,1000],[10,60,1000]]
	samples_size=10000
	handles=[]
	labels=[]
	bins=25
	print("Uniform Dist Vs Normal Dist")
	print("=="*20)

	for a,b,n in params:
		samples=[]
		for i in range(samples_size):
			sample=stats.uniform.rvs(a,b-a,size=n)
			samples.append(np.sum(sample))
		normalized_uniform_samples=[(x-n*(b+a)/2)/np.sqrt(n*(b-a)**2/12) for x in samples]
		hist,bin_edges=np.histogram(normalized_uniform_samples,bins=bins,density=True)
		handle,=subplotAxes.plot(bin_edges[1:],hist)
		handles.append(handle)
		labels.append('a=%d,b=%d,n=%d'%(a,b,n))

		print("a=%d,b=%d,n=%d" %(a,b,n))
		print(stats.describe(normalized_uniform_samples))
		print(stats.ttest_1samp(normalized_uniform_samples,0))
		print(stats.kstest(normalized_uniform_samples,'norm'))
		print("-------------------")
	subplotAxes.legend(handles=handles,labels=labels,loc='best',frameon=False,labelspacing=0)
	drawNormalDistPdf(subplotAxes)
	subplotAxes.set_title('Uniform Dist Vs Normal Dist')

def normalizedExponentialDistVsNormalDist(subplotAxes):
	params=[[0.6,100],[0.6,1000],[3,100],[3,1000]]
	sample_size=10000 #bigger sample_size can reduce the sawtooth while it takes much more time
	handles=[]
	labels=[]
	bins=25
	print("Exponential Dist Vs Normal Dist")
	print("=="*20)

	for lamb,n in params:
		samples=[]
		for i in range(sample_size):
			sample=stats.expon.rvs(scale=lamb,size=n)
			samples.append(np.sum(sample))
		normalized_expon_samples=[(x-n*lamb)/(np.sqrt(n)*lamb) for x in samples]
		hist,bin_edges=np.histogram(normalized_expon_samples,bins,density=True)
		handle,=subplotAxes.plot(bin_edges[1:],hist,label=r"$\lambda$=%.2f,n=%d" %(lamb,n))
		handles.append(handle)

		print("lambda=%.2f, n=%d" %(lamb,n))
		print(stats.describe(normalized_expon_samples))
		print(stats.ttest_1samp(normalized_expon_samples,0))
		print(stats.kstest(normalized_expon_samples,'norm'))
		print("----------------------")
	subplotAxes.legend(handles=handles,loc='best',frameon=False,labelspacing=0)
	drawNormalDistPdf(subplotAxes)
	subplotAxes.set_title('Exponential Dist Vs Norm Dist')



def drawSomePmf():
	fig1=plt.figure(1,figsize=(15,15),clear=True) 
	gs=gridspec.GridSpec(2, 2,left=0.1,hspace=0.3)

	normalizedBinomDistVsNormalDist(plt.subplot(gs[0,0]))
	normalizedPossionDistVsNormalDist(plt.subplot(gs[0,1]))
	normalizedUniformDistVsNormalDist(plt.subplot(gs[1,0]))
	normalizedExponentialDistVsNormalDist(plt.subplot(gs[1,1]))

	plt.show()

if __name__ == '__main__':
	drawSomePmf()