import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm # colormap
from matplotlib.ticker import LinearLocator,FormatStrFormatter
from matplotlib import colors
# from scipy.stats import uniform


def baseUseOfAxes3D(ax):
	x,y=np.mgrid[-10:10:101j,-8:8:101j]
	# x=np.arange(-10,10, 0.4)
	# y=np.arange(-10, 10,0.4)
	# x,y=np.meshgrid(x,y)
	r=np.sqrt(x**2+y**2)
	z=np.sin(r)
	# print(z.shape)
	ax.plot_surface(x,y,z,rstride=1,cstride=1) # rstride and cstride are used to set the sampling step in each direction, they are mutually exclusive with rcount and ccount
	# ax.plot_surface(x,y,z,rcount=3,ccount=3)#rcount and ccount are used to set the maximum number of samples used in each direction, Defaults to 50
	# ax.plot3D(x,y)
	ax.contourf(x,y,z,zdir='z',offset=-2)
	plt.show()

def drawBall(ax):
	r=5

	# Method 1
	# theta,phi=np.mgrid[0:2*np.pi:200j,0:np.pi:100j]
	# 
	# z=r*np.cos(phi)
	# print(z.shape, ...)
	# x=r*np.sin(phi)*np.cos(theta)
	# y=r*np.sin(phi)*np.sin(theta)

	# Method 2: This way can reduce storage memory
	theta=np.linspace(0, 2*np.pi,30)
	phi=np.linspace(0, np.pi,20)
	z=r*np.outer(np.cos(phi), np.ones(theta.shape))
	x=r*np.outer(np.sin(phi),np.cos(theta))
	y=r*np.outer(np.sin(phi), np.sin(theta))

	ax.plot_surface(x,y,z,)
	# ax.plot_wireframe(x,y,z)
	ax.contourf(x,y,z,zdir='z',offset=-7)

def draw3DCurve(ax):
	mpl.rcParams['legend.fontsize']=10
	theta=np.linspace(-3*np.pi, 3*np.pi,100)
	z=np.linspace(-2, 2,100)
	r=z**2+1
	x=r*np.sin(theta)
	y=r*np.cos(theta)
	ax.plot(x,y,z,label='3D cureve')
	# ax.plot(x,y,zdir='z',label='2D curve') # 2D graph
	ax.legend(frameon=False)

def drawScatter(ax):
	n=100
	data_fmt=[('r','o',-15,15),('b','^',5,25)]
	for c,m,zhigh,zlow in data_fmt:
		xs=np.random.uniform(23,25,n)
		ys=np.random.uniform(-10,50,n)
		zs=np.random.uniform(zlow,zhigh,n)
		ax.scatter(xs,ys,zs,c=c,marker=m,depthshade=True)

	ax.set_xlabel('X label')
	ax.set_ylabel('Y label')
	ax.set_zlabel('Z label')	

def drawWireFrame(ax):
	x,y,z=axes3d.get_test_data(0.1)
	ax.plot_wireframe(x,y,z,rcount=10,ccount=10)

def drawSurfacePlot(ax):
	x,y=np.meshgrid(np.arange(-5,5,0.05),np.arange(-5,5, 0.05))
	r=np.sqrt(x**2+y**2)
	z=np.sin(r)

	surf=ax.plot_surface(x,y,z,cmap=cm.rainbow,linewidth=0,rcount=50,ccount=50)

	ax.set_zlim(-1.01,1.01)
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	fig.colorbar(surf,shrink=0.5,aspect=8)

def drawSurfaceWithFaceColors(ax):
	x,y=np.mgrid[-5:5:40j,-5:5:40j]
	z=np.sin(np.sqrt(x**2+y**2))

	colortuple=('r','g','y','b')
	colors=np.empty(z.shape,dtype=str)
	for i in range(len(x)):
		for j in range(len(y)):
			colors[i,j]=colortuple[(i+j)%len(colortuple)]
			

	surf=ax.plot_surface(x,y,z,facecolors=colors,linewidth=0)

def drawTriangleSurface(ax):
	n_radii=8
	n_angles=36

	radii=np.linspace(0.125,1.0,n_radii)
	angles=np.linspace(0, 2*np.pi,n_angles,endpoint=False)

	angles=np.repeat(angles[:,np.newaxis], n_radii,axis=1)

	x=np.append(0, (radii*np.cos(angles)).flatten())
	y=np.append(0, (radii*np.sin(angles)).flatten())
	print(x.shape, y.shape)

	z=np.sin(-x*y)

	colortuple=['r','g','b','y','m']
	colors=np.empty(z.shape)

	for i in range(len(x)):
		colors[i]=colortuple[i%len(colortuple)]

	surf=ax.plot_trisurf(x,y,z,cmap=cm.rainbow,linewidth=0,antialiased=True)

	# fig.colorbar(surf,shrink=0.5)

def drawTriangleSurfacesWithTriang():
	import matplotlib.tri as mtri
	fig=plt.figure(2)

	# First Plot
	u=np.linspace(0, 2*np.pi,endpoint=True,num=50)
	v=np.linspace(-0.5,0.5, endpoint=True,num=10)

	u,v=np.meshgrid(u,v)
	u,v=u.flatten(),v.flatten()

	x=(1+0.5*v*np.cos(u/2.0))*np.cos(u)
	y=(1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
	z=0.5 * v * np.sin(u / 2.0)

	tri=mtri.Triangulation(u,v)

	ax=fig.add_subplot(121,projection='3d')
	ax.plot_trisurf(x,y,z,triangles=tri.triangles,cmap=plt.cm.Spectral)
	ax.set_zlim(-1,1)

	# Second Plot
	n_angles = 36
	n_radii = 8
	min_radius = 0.25

	radii=np.linspace(min_radius, 0.95,n_radii)

	angles=np.linspace(0,2*np.pi, n_angles, endpoint=False)
	angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

	angles[:,1::2] += np.pi/n_angles

	x=(radii * np.cos(angles)).flatten()
	y=(radii * np.sin(angles)).flatten()
	z=(np.cos(radii) * np.cos(angles * 3.0)).flatten()

	triang=mtri.Triangulation(x, y)

	# mask off unwanted triangles
	xmid = x[triang.triangles].mean(axis=1)
	ymid = y[triang.triangles].mean(axis=1)
	mask = np.where(xmid**2 + ymid**2 < min_radius**2, 1, 0)
	triang.set_mask(mask)

	ax=fig.add_subplot(122,projection='3d')
	ax.plot_trisurf(triang,z,cmap=cm.CMRmap)

def drawContour3d(ax):
	x,y,z=axes3d.get_test_data(0.05)
	cset=ax.contour(x,y,z,cmap=cm.coolwarm,zdir='z',extend3d=True)
	ax.clabel(cset,fontsize=19,inline=1)

def drawContourOn3Dim(ax):
	x,y,z=axes3d.get_test_data(0.05)
	

	min_values=[x.min(),y.min(),z.min()]
	max_values=[x.max(),y.max(),z.max()]
	min_3=[m*(1-0.2*(m>0 and 1 or -1)) for m in min_values]
	max_3=[m*(1+0.2*(m>0 and 1 or -1))for m in max_values]

	ax.plot_surface(x,y,z,alpha=0.5)

	ax.contour(x,y,z,zdir='z',offset=min_3[2],cmap=cm.coolwarm)
	ax.contour(x,y,z,zdir='x',offset=min_3[0],cmap=cm.coolwarm)
	ax.contour(x,y,z,zdir='y',offset=max_3[1],cmap=cm.coolwarm)

	ax.set_xlabel('X')
	ax.set_xlim(min_3[0],max_3[0])
	ax.set_ylabel('Y')
	ax.set_ylim(min_3[1],max_3[1])
	ax.set_zlabel('Z')
	ax.set_zlim(min_3[2],max_3[2])

def drawContourf3d(ax):
	x,y,z=axes3d.get_test_data(0.05)
	ax.contourf(x,y,z,cmap=cm.coolwarm)

def drawContourfOn3Dim(ax):
	x,y,z=axes3d.get_test_data(0.05)
	min_values=[x.min(),y.min(),z.min()]
	max_values=[x.max(),y.max(),z.max()]
	min_3=[m*(1-0.2*(m>0 and 1 or -1)) for m in min_values]
	max_3=[m*(1+0.2*(m>0 and 1 or -1)) for m in max_values]

	ax.plot_surface(x,y,z,alpha=0.5,rcount=50,ccount=50)
	ax.contourf(x,y,z, zdir='x', offset=min_3[0],cmap=cm.coolwarm)
	ax.contourf(x,y,z, zdir='y', offset=max_3[1],cmap=cm.coolwarm)
	ax.contourf(x,y,z, zdir='z', offset=min_3[2],cmap=cm.coolwarm)

	ax.set_xlabel('X')
	ax.set_xlim(min_3[0],max_3[0])
	ax.set_ylabel('Y')
	ax.set_ylim(min_3[1],max_3[1])
	ax.set_zlabel('Z')
	ax.set_zlim(min_3[2],max_3[2])

def cc(arg):
	return colors.to_rgba(arg,alpha=0.6)

def drawPolys3D(ax):
	from matplotlib.collections import PolyCollection
	
	xs=np.arange(0, 10,0.4)
	verts=[]
	zs=[0.0,1.0,2.0,3.0]
	for z in zs:
		ys=np.random.rand(len(xs))
		ys[0],ys[-1]=0,0
		verts.append(list(zip(xs,ys)))

	poly=PolyCollection(verts,facecolors=[cc('r'),cc('g'),cc('b'),cc('y')])
	poly.set_alpha(0.7)

	ax.add_collection3d(poly,zs=zs,zdir='y')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	ax.set_xlim(0,10)
	ax.set_ylim(-1,4)
	ax.set_zlim(0,1)

def drawBars(ax):
	for c,z in zip(['r','g','b','y'],[30,20,10,0]):
		xs=np.arange(20) # xs is the x coordinates of the left sides of the bars; 
		ys=np.random.rand(20) # ys is the height of the bars

		cs = [c] * len(xs)
		ax.bar(xs,ys,zs=z,zdir='y',color=cs,alpha=0.8) #  zs is z coordinate of bars

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

def drawQuiver(ax):
	# The x, y and z coordinates of the arrow locations, (default is tail of arrow)
	x,y,z=np.meshgrid(np.arange(-0.8, 1,0.2),np.arange(-0.8, 1,.2),np.arange(-0.8, 1,0.8))

	# u v w are The x, y and z components of the arrow vectors
	u=np.sin(np.pi*x) * np.cos(np.pi*y) * np.cos(np.pi*z)
	v=-np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
	w=(np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y ) * np.sin(np.pi * z))

	# length: The length of each quiver, default to 1.0, the unit is the same with the axes
	# arrow_length_ratio: The ratio of the arrow head with respect to the quiver, default to 0.3
	# normalize: When True, all of the arrows will be the same length. This defaults to False, where the arrows will be different lengths depending on the values of u,v,w.
	ax.quiver(x,y,z,u,v,w,length=0.1, arrow_length_ratio=0.3,normalize=True) 
	
def draw2dCollections3d(ax):
	# plot a sin curve using the x and y axes
	x=np.linspace(0, 1,100)
	y=np.sin(x * 2 * np.pi) / 2 + 0.5
	ax.plot(x,y,zs=0,zdir='z', label='curve in (x,y)') # the plot's params aren't the same with its 2d version

	# plot scatterplot data on the x and z axes
	colors=('r','g','b','k')
	x=np.random.sample(20*len(colors))
	y=np.random.sample(20*len(colors))
	c_list=[]

	for c in colors:
		# c_list.append([c]*20) # add a list object (as an element) to c_list, get a list with 2 dimension
		c_list.extend([c]*20) # add a list to c_list , get a list with 1 dimension
		# c_list+=[c] * 20 # same as extend 

	# By using zdir='y', the y value of these points is fixed to the zs value 0
	# and the (x,y) points are plotted on the x and z axes.
	# ax.scatter(x,y,zs=0,zdir='y',c=np.array(c_list).reshape(80,),label='points in (x,z)')
	ax.scatter(x,y,zs=0,zdir='y',c=c_list,label='points in (x,z)')

	ax.legend()
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_zlim(0,1)

	ax.view_init(elev=20,azim=-35)

def drawText3D(ax):
	zdirs=(None,'x','y','z',(1,1,0),(1,1,1))
	xs=(1,4,4,9,4,1)
	ys=(2,5,8,10,1,2)
	zs=(10,3,8,9,1,8)

	for zdir, x, y, z in zip(zdirs,xs,ys,zs):
		label='(%d, %d, %d), dir=%s' % (x,y,z,zdir)
		ax.text(x,y,z,label,zdir)

	ax.text(9,0,0,'red',color='red')

	ax.text2D(0.05,0.95,'2D Text', transform=ax.transAxes)

	ax.set_xlim(0,10)
	ax.set_ylim(0,10)
	ax.set_zlim(0,10)
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')


def f(t):
	s1=np.cos(2*np.pi*t)
	e1=np.exp(-t)
	return np.multiply(s1, e1)

def mixedSubplots():

	#First subplot
	t1=np.arange(0.0,5.0, 0.1)
	t2=np.arange(0.0,5.0, 0.02)
	t3=np.arange(0.0,2.0, 0.01)

	fig=plt.figure(figsize=plt.figaspect(2.))
	fig.suptitle('A table of 2 subplots')
	ax=fig.add_subplot(211)
	l=ax.plot(t1,f(t1),'bo',t2,f(t2),'k--',markerfacecolor='green')
	ax.grid(True)
	ax.set_ylabel('Damped oscillation')

	# Second subplot
	ax=fig.add_subplot(212, projection='3d')
	x=np.arange(-5, 5,0.25)
	xlen=len(x)
	y=np.arange(-5, 5,0.25)
	ylen=len(y)
	x,y=np.meshgrid(x,y)
	r=np.sqrt(x**2 + y**2)
	z=np.sin(r)

	surf=ax.plot_surface(x,y,z,ccount=50,rcount=50,linewidth=0,antialiased=False)

	ax.set_zlim3d(-1,1)



if __name__ == '__main__':
	fig=plt.figure()

	# ax=Axes3D(fig) # older version could use this way to create a 3D axes
	# ax=fig.add_subplot(111,projection='3d') # new version
	ax=fig.gca(projection='3d') # 3rd way

	# baseUseOfAxes3D(ax)
	# drawBall(ax)
	# draw3DCurve(ax)
	# drawScatter(ax)
	# drawWireFrame(ax)
	# drawSurfacePlot(ax)
	# drawSurfaceWithCustomedColors(ax)
	# drawTriangleSurfaces(ax)
	# drawTriangleSurfacesWithTriang()
	# drawContour3d(ax)
	# drawContourOn3Dim(ax)
	# drawContourf3d(ax)
	# drawContourfOn3Dim(ax)
	# drawPolys3D(ax)
	# drawBars(ax)
	# drawQuiver(ax)
	# draw2dCollections3d(ax)
	# drawText3D(ax)
	mixedSubplots()

	plt.show()