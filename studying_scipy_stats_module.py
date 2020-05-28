import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy import stats

def plotNormalDistPdf(subplotAxes):	
	x1=np.linspace(-10,10,200)
	x2=np.linspace(-7,13,200)
	x3=np.linspace(-13,7,200)
	standardNormalPdf=norm.pdf(x1)
	normal_loc3_scale10=norm.pdf(x2,loc=3,scale=3)
	normal_locN3_scale5=norm.pdf(x3,loc=-3,scale=0.8)
	handle1,handle2,handle3=subplotAxes.plot(x1,standardNormalPdf,x2,normal_loc3_scale10,x3,normal_locN3_scale5)
	legends=subplotAxes.legend(handles=[handle1,handle2,handle3],labels=[r'$\mu=0,\sigma=1$',r'$\mu=3,\sigma=3$',r'$\mu=-3,\sigma=0.8$'],frameon=False)
	subplotAxes.set_title("Normal Distributions' pdf")

def plotChiSquareDistPdf(subplotAxes):
	dfs=[1,2,4,6,11]
	handles=[]
	labels=[]
	for df in dfs:
		x=np.linspace(stats.chi2.ppf(0.001,df), stats.chi2.ppf(0.999,df),1000)
		chiSquarePdf=stats.chi2.pdf(x,df)
		handle,=subplotAxes.plot(x,chiSquarePdf)
		handles.append(handle)
		labels.append('n=%d' %df)
	subplotAxes.set_ylim(0,0.5)
	subplotAxes.set_xlim(0,15)
	subplotAxes.legend(handles=handles,labels=labels,frameon=False)
	subplotAxes.set_title(r"$\chi^2$ Distributions' pdf")



def plotTDistPdf(subplotAxes):
	freedoms=[1,2,4,100]
	handles=[]
	labels=[]
	x_lower=[]
	x_upper=[]
	for freedom in freedoms:
		x=np.linspace(stats.t.ppf(0.001,freedom), stats.t.ppf(0.999,freedom),10000)
		x_lower.append(np.min(x))
		x_upper.append(np.max(x))
		tPdf=stats.t.pdf(x,freedom)
		handle,=subplotAxes.plot(x,tPdf)
		handles.append(handle)
		labels.append("n=%d" %freedom)

	# normal_x=np.linspace(np.min(x_lower),np.max(x_upper),1000)
	normal_x=np.linspace(stats.norm.ppf(0.001), stats.norm.ppf(0.999),10000)
	normal_pdf=stats.norm.pdf(normal_x)
	handle,=subplotAxes.plot(normal_x,normal_pdf,linewidth=4,alpha=0.7)
	handles.append(handle)
	labels.append('normal dist')

	subplotAxes.legend(handles=handles,labels=labels,frameon=False)
	subplotAxes.set_title("T Distributions' pdf")
	# subplotAxes.set_xlim(np.median(x_lower),np.median(x_upper))
	# subplotAxes.set_xlim(np.mean(x_lower),np.mean(x_upper))
	subplotAxes.set_xlim(-5,5)

def plotFDistPdf(subplotAxes):
	freedoms=[[3,40],[7,30],[15,20],[20,15],[30,10],[40,3]]
	handles=[]
	labels=[]
	x_upper=0
	y_upper=0
	for freedom in freedoms:
		x=np.linspace(0, stats.f.ppf(0.99,freedom[0],freedom[1]),1000)
		if np.max(x)>x_upper:
			x_upper=np.max(x)
		fPdf=stats.f.pdf(x,freedom[0],freedom[1])
		if np.max(fPdf)>y_upper:
			y_upper=np.max(fPdf)
		handle,=subplotAxes.plot(x,fPdf)
		handles.append(handle)
		labels.append('n1=%d, n2=%d' %(freedom[0],freedom[1]))

	subplotAxes.legend(handles=handles,labels=labels,frameon=False)
	subplotAxes.set_title("F Distributions' pdf")
	subplotAxes.set_xlim(0,6)
	subplotAxes.set_ylim(0,y_upper*1.1)


def drawSomePdf():
	fig1=plt.figure(1,figsize=(15,15),clear=True)
	gs=gridspec.GridSpec(2, 2)
	normDistSubplot=plt.subplot(gs[0,0])
	chiSquareDistSubplot=plt.subplot(gs[0,1])
	tDistSubplot=plt.subplot(gs[1,0])
	fDistSubplt=plt.subplot(gs[1,1])

	plotNormalDistPdf(normDistSubplot)
	plotChiSquareDistPdf(chiSquareDistSubplot)
	plotTDistPdf(tDistSubplot)
	plotFDistPdf(fDistSubplt)

	plt.show()

if __name__ == '__main__':
	drawSomePdf()


