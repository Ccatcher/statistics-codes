# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
'''
1. Simulating some common distribution's probability density curve with numpy's random generator
2. Mark down some technique about drawing a graph with matplotlib
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def generateNormalDistSample(mu,sigma,size=None):
	normalSamples=np.random.normal(mu,sigma,size)
	return normalSamples

def generateChiSquareDistSample(freedomDegrees,size=None):
	chiSquareSamples=np.random.chisquare(freedomDegrees,size)
	return chiSquareSamples

def generateTDistSample(freedomDegrees,size=None):
	TSamples=np.random.standard_t(freedomDegrees,size)
	return TSamples

def generateFDistSampel(freedomDegree1,freedomDegree2,size=None):
	FSamples=np.random.f(freedomDegree1, freedomDegree2,size)
	return FSamples


def normalFun(x,mu,sigma):
	pdf=np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
	return pdf

def drawNormalDistProbabilityDensity(subplotAxes):
	standardNormalSamples=generateNormalDistSample(0, 1,100000)
	normalSamples_sigma5=generateNormalDistSample(0, 2,100000)
	normalSamples_sigma10=generateNormalDistSample(0, 4,100000)
	normalSamples=np.array([standardNormalSamples,normalSamples_sigma5,normalSamples_sigma10])
	NF1=np.histogram(standardNormalSamples,15,density=True)#density用于将落在某一范围内的样本值转换为频率值
	NF2=np.histogram(normalSamples_sigma5,15,density=True)
	NF3=np.histogram(normalSamples_sigma10,50,density=True)
	
	subplotAxes.set_xlim(-10,10) #设置x轴的范围
	# subplotAxes.text(4,0.3,r'----$\mu=0,\sigma=1,\chi^2=1$',bbox=dict(facecolor='red',alpha=0.5),verticalalignment='baseline',animated=False)
	n1,n2,n3=subplotAxes.plot(NF1[1][1:], NF1[0],'k-',NF2[1][1:], NF2[0],'k--',NF3[1][1:], NF3[0],'k-.') #可以使用本行或者下面三行代码获取绘图对象，便于后续的添加图例等操作
	# n1,=subplotAxes.plot(NF1[1][1:],NF1[0],'k-',label=r'$\mu=0,\sigma=1$',)
	# n2,=subplotAxes.plot(NF2[1][1:],NF2[0],'r--')
	# n3,=subplotAxes.plot(NF3[1][1:],NF3[0],'g-.')

	fontLegend={'family':'Times New Roman','weight':'normal','size':10,'style':'italic'}

	#添加图例的方法有很多参数，下面一行展示了可调的一些参数
	# legend=subplotAxes.legend(handles=[n1,n2,n3],labels=[r'$\mu=0,\sigma=1$',r'$\mu=0,\sigma=2$',r'$\mu=0,\sigma=4$'],prop=fontLegend,loc=(0.6,0.6),fontsize='xx-small',handlelength=3)
	legend=subplotAxes.legend(handles=[n1,n2,n3],labels=[r'$\mu=0,\sigma=1$',r'$\mu=0,\sigma=2$',r'$\mu=0,\sigma=4$'],prop=fontLegend,loc='upper right',frameon=False,labelspacing=0,handlelength=3)
	# # subplotAxes.hist(standardNormalSamples,100,density=True)

	xlabelFont={
	'size':'large',
	'weight':'normal',
	'style':'italic'
	}
	ylabelFont={
	# 'rotation':'horizontal',# or set to some angle in degrees
	'size':'large',
	'weight':'normal',
	'style':'italic'
	}
	subplotAxes.set_xlabel('x',fontdict=xlabelFont)
	subplotAxes.set_ylabel('p',ylabelFont)

	xtickLabel={
	'size':10,
	}
	ytickLabel={
	'size':10
	}

	subplotAxes.set_title("Normal distribution samples")
	
	# subplotAxes.tick_params(axis='x',direction='in',color='r',pad=5,labelsize=9,labelrotation=9)
	subplotAxes.tick_params(axis='x',pad=5,labelsize=9,labelrotation=9)
	# 另一种实现修改坐标轴刻度及标签的方法
	# tickLabels=subplotAxes.get_xticklabels()+subplotAxes.get_yticklabels()
	# [label.set_fontsize(8) for label in tickLabels]
	# [label.set_fontstyle('italic') for label in tickLabels]

# Draw box-plot with standard normal distribution samples
def drawBoxChartOfNormalDist(subplotAxes):
	standardNormalSamples=generateNormalDistSample(0, 1,20000)
	subplotAxes.boxplot(standardNormalSamples,vert=False)
	
def drawChiSquareDistProbabilityDensity(ax):
	freedoms=[1,2,4,6,11]
	handles=[]
	for freedom in freedoms:
		samples=generateChiSquareDistSample(freedom,100000)
		hist,binEdge=np.histogram(samples,50,density=True)

		handle,=ax.plot(binEdge[0:-1],hist)
		handles.append(handle)

	ax.set_xlim(0,15)
	ax.set_ylim(0,0.5)
	ax.set_title(r'$\chi^2$ distribution samples')
	ax.legend(handles,labels=['n=1','n=2','n=4','n=6','n=11'],frameon=False,labelspacing=0)

def drawTDistProbabilityDens(ax):
	freedoms=[1,3,7,1000]
	linStyles=['k-.','k:','k--','k-']
	handles=[]
	for i in range(4):
		samples=generateTDistSample(freedoms[i],100000)
		hist,binEdge=np.histogram(samples,np.linspace(-5,5,50),density=True)#这里的bins参数如果使用int值，存在异常值偏大导致bin的区间过大，结果不准确，故限定各个bin的范围
		
		# print(binEdge, ...)
		handle,=ax.plot(binEdge[:-1],hist,linStyles[i])
		handles.append(handle)

	ax.set_xlim(-5,5)
	ax.set_title('t distribution samples',fontstyle='italic')
	ax.legend(handles,labels=['n=1','n=3','n=7',r'n$\to\infty$'],frameon=False)

def drawFDistProbDens(ax):
	freedomNumS=[3,10,11]
	freedomDens=[11,40,3]
	linStyles=['k-.','k--','k-']
	handles=[]
	for i in range(3):
		samples=generateFDistSampel(freedomNumS[i], freedomDens[i],100000)
		hist,binEdge=np.histogram(samples,np.linspace(0, 6,50),density=True)
		handle,=ax.plot(binEdge[0:-1],hist,linStyles[i],label='n1='+str(freedomNumS[i])+', n2='+str(freedomDens[i]))

		handles.append(handle)

	ax.set_title('F distribution samples')
	ax.set_xlim(0,6)
	ax.set_ylim(0,1)
	ax.legend(frameon=False)

def drawSomeDistributionCurves():
	fig1=plt.figure(1,figsize=(15,15),clear=True) #若绘制多幅图形，可以设置第一个参数作为不同图形的id

	gs=gridspec.GridSpec(2, 2,wspace=0.15,hspace=0.35)
	# fig1.subplots_adjust(left=0.05,wspace=0.2,hspace=0.4)
	normalDistSubplot=plt.subplot(gs[0,0])
	chiSquareDistSubplot=plt.subplot(gs[0,1])
	tDistSubplot=plt.subplot(gs[1,0])
	fDistSubplt=plt.subplot(gs[1,1])
	drawNormalDistProbabilityDensity(normalDistSubplot)
	drawChiSquareDistProbabilityDensity(chiSquareDistSubplot)
	drawTDistProbabilityDens(tDistSubplot)
	drawFDistProbDens(fDistSubplt)

	# drawBoxChartOfNormalDist(chiSquareDistSubplot)

	plt.show()

if __name__=='__main__':
	drawSomeDistributionCurves()

# def normalDis(x):
	
# 	x.sort()
# 	y=[x[0]]
# 	z=[0]
# 	for ix in x:
# 		i=0
# 		for iy in y:
# 			if abs(ix-iy)<0.1:
# 				z[i]+=1.0/10000
# 				break
# 			else:
# 				i+=1
# 		if i>=len(y):
# 			y.append(ix)
# 			z.append(0)
# 	plt.plot(y,z)
# 	plt.xlim(-5,5)
# 	plt.show()

# x=np.random.randn(10000)
# # x1=np.random.randn(10000)
# # x2=np.random.randn(10000)
# # x3=np.random.randn(10000)
# # x4=np.random.randn(10000)
# x5=np.random.randn(10000)
# x6=np.random.randn(10000)
# y=np.random.randn(10000)
# cc=x6**2+y**2+x5**2
# # z=x3**2+x4**2+x5**2+x6**2+x**2+x1**2+x2**2+y**2
# zt=x/np.sqrt(cc)
# normalDis(zt)
# x=np.arange(-15,15,0.001)
# x=np.linspace(-15, 15,1000)
# y=normalFun(x, 0, 1)
# sump=0
# alpha=0
# for p in y[::-1]:
# 	sump+=p
# 	alpha+=1
# 	if sump>=50:		
# 		break

# print(x[-alpha], ...)
# plt.plot(x,y)
# plt.show()