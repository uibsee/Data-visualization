# 001 Introduction to Matplotlib and basic line
from matplotlib import pyplot as plt
x=[1,2,3,4]
y=[5,6,7,8]
plt.figure()  # 선택과정
plt.plot(x,y)
plt.title('matplotlib gallery') # 선택과정
plt.savefig('plot01.png', dpi=200) # 선택과정
plt.show()

# 002 basic line : y값만 지정하기 (x의 값은 0부터 정수로 지정된다.)
from matplotlib import pyplot as plt
plt.plot([1,3,2,4,7,-1,5])
plt.show()

# 003-1 basic line : 리스트를 이용해 간단한 함수 그리기
from matplotlib import pyplot as plt
x=[x/10 for x in range(100)]
plt.plot(x, [x**2-10*x for x in x])
plt.show()

# 003-2 basic line : Numpy로 이용해 간단한 함수 그리기
from matplotlib import pyplot as plt
import numpy as np
x=np.arange(0.0, 10.0, 0.01)
plt.plot(x, [x**2-10*x for x in x])
plt.show()

# 004 basic line : 여러 개의 그래프 같이 그리기
from matplotlib import pyplot as plt
x=[x/10 for x in range(100)]
plt.plot(x, [x**2-10*x for x in x], x, [x/2-8 for x in x])
plt.show()

# 005-1 Grid, Axis, Legends, Titles, and Labels with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,5, 0.1)
plt.plot(x, x**2, label='First Line')
plt.plot(x, x**0.5, label='Second Line')
plt.grid(True)
plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.title('Interesting Graph\nCheck it out')
plt.legend(loc='upper left')
plt.show()

# 005-2 Grid, Axis, Legends, Titles, and Labels with Matplotlib : 수식 사용하기($\수식내용$)
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,5, 0.1)
plt.plot(x, x**2, label='First Line')
plt.plot(x, x**0.5, label='Second Line')
plt.grid(True)
plt.xlabel('$\pi$')
plt.ylabel(r'$\alpha>\beta$')
plt.title('Interesting Graph\nCheck it out')
plt.legend(loc='upper left')
plt.show()

# 006-1 Graph style with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3,0.3)
plt.plot(y,'cx--',y+1,'mo:',y+2,'kp-.')
plt.show()

# 006-2 Graph style2 with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3,0.3)
plt.plot(y, color='blue', linestyle='dashdot', linewidth=4, marker='o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=3, markersize=12)
plt.show()

# 007 tick setting(축에 눈금 표시 하기)
import matplotlib.pyplot as plt
x = [5, 3, 7, 2, 4, 1]
plt.plot(x)
plt.xticks(range(len(x)), ['a', 'b', 'c', 'd', 'e', 'f'])
plt.yticks(range(1, 8, 2))
plt.show()

# 008-1 Histograms with Matplotlib
import matplotlib.pyplot as plt
population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]
plt.hist(population_ages)
plt.show()

# 008-2 Histograms with Matplotlib
import matplotlib.pyplot as plt
population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]
plt.hist(population_ages, bins=4)
plt.show()

# 008-3 Histograms with Matplotlib
import matplotlib.pyplot as plt
population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]
bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]
plt.hist(population_ages, bins)
plt.show()

# 009-1 Error bar charts with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,4,0.2)
y=y=np.exp(-x)
e1=0.1*np.abs(np.random.randn(len(y)))
plt.errorbar(x,y,yerr=e1,fmt='-.')
plt.show()

# 009-2 Error bar charts with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,4,0.2)
y=y=np.exp(-x)
e1=0.1*np.abs(np.random.randn(len(y)))
e2=0.1*np.abs(np.random.randn(len(y)))
plt.errorbar(x,y,yerr=e1,xerr=e2,fmt='-.', capsize=5)
plt.show()

# 009-3 Error bar charts with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,4,0.2)
y=y=np.exp(-x)
e1=0.1*np.abs(np.random.randn(len(y)))
e2=0.1*np.abs(np.random.randn(len(y)))
plt.errorbar(x,y,yerr=[e1,e2],fmt='-.', ecolor='r')
plt.show()

# 010-1 Bar Charts with Matplotlib
import matplotlib.pyplot as plt
plt.bar([1,2,3],[3,2,5])
plt.show()

# 010-2 Bar Charts with Matplotlib
import matplotlib.pyplot as plt
x1=[1,2,3,4]
x2=['A','B','C','D']
y=[40,70,30,85]
plt.bar(x1,y)
plt.xticks(x1,x2)
plt.yticks(y)
plt.show()

# 010-3 Bar Charts with Matplotlib
import matplotlib.pyplot as plt
plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")
plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')
plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')
plt.title('Epic Graph\nAnother Line! Whoa')
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.show()

# 011 Pie Charts with Matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
slices = [7,2,2,13]
activities = ['sleeping','eating','working','playing']
plt.pie(slices,
        labels=activities,
        startangle=90,
        shadow= True,
        explode=(0,0.1,0,0),
        autopct='%1.1f%%')
plt.legend(loc=1)
plt.title('Interesting Graph\nCheck it out')
plt.show()

# 012-1 Scatter Plots with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x,y, label='test1', color='c', s=25, marker="o")
plt.legend()
plt.show()

# 012-2 Scatter Plots with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x,y, label='test1', color='c', s=25, marker="o")
plt.scatter(y,x, label='test2', color='b', s=25, marker="x")
plt.legend()
plt.show()

# 012-3 Scatter Plots with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(300)
y = np.random.randn(300)
size=50*np.random.randn(300)
plt.scatter(x,y, s=size, marker="o", alpha=0.7)
plt.show()

# 013 Stack Plots with Matplotlib
import matplotlib.pyplot as plt
days=[1,2,3,4,5]
sleeping=[7,8,6,11,7]
eating=[2,3,4,9,2]
working=[7,8,4,2,5]
playing=[3,5,7,8,13]
labels=['sleeping', 'eating', 'working', 'playing']
plt.stackplot(days, sleeping,eating,working,playing, labels=labels)
plt.legend(loc=2)
plt.show()

# 014-1 polar charts with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
theta=np.arange(0,2,1/180)*np.pi
plt.polar(3*theta,theta/5)
plt.polar(theta,np.cos(4*theta))
plt.show()

# 014-2 polar charts with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
theta=np.arange(0,2,1/180)*np.pi
r=np.sin(theta)
plt.polar(theta,r)
plt.thetagrids(range(45,360,90))
plt.rgrids(np.arange(0.5,3.0,0.5), angle=0)
plt.show()

# 015 subplots with Matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(211)
ax1.plot([1,2,3],[2,3,4])
ax2=fig.add_subplot(212)
ax2.plot([1,2,3],[3,2,1])
plt.show()

# 016-1 Logarithmic axes with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,20,0.01)
y=np.cos(np.pi*x)
plt.semilogx(x,y)
plt.show()

# 016-2 Logarithmic axes with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,20,0.01)
y=x**2
plt.semilogy(x,y)
plt.show()


# 016-3 Logarithmic axes with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,20,0.01)
y=np.cos(np.pi*x)
plt.loglog(x,y)
plt.show()


# 017-1 Contour plots with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(-2,2,0.01)
y=np.arange(-2,2,0.01)
X,Y=np.meshgrid(x,y)
Z=X*X/9+Y*Y/4-1
cs=plt.contour(Z)
plt.clabel(cs)
plt.show()


# 017-2 Contour plots with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(-2,2,0.01)
y=np.arange(-2,2,0.01)
X,Y=np.meshgrid(x,y)
Z=X*X/9+Y*Y/4-1
plt.contourf(Z)
plt.colorbar()
plt.show()

#18 3D Curve with Matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
fig=plt.figure()
ax=Axes3D(fig)
ax.plot(x,y,z)
plt.show()

#19-1 3D Subface with Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
x=np.arange(-5,5,0.01)
y=np.arange(-5,5,0.01)
X,Y=np.meshgrid(x,y)
R=np.sqrt(X**2+Y**2)
Z=np.sin(R)
fig=plt.figure()
ax=Axes3D(fig)
ax.plot_surface(X,Y,Z,cmap=cm.viridis)
plt.show()

#19-2 3D Wireframe with Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
x=np.arange(-5,5,0.01)
y=np.arange(-5,5,0.01)
X,Y=np.meshgrid(x,y)
R=np.sqrt(X**2+Y**2)
Z=np.sin(R)
fig=plt.figure()
ax=Axes3D(fig)
ax.plot_wireframe(X,Y,Z)
plt.show()

#20 3D Scatter with Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(df['X'], df['Y'], df['Z'],c='skyblue',marker='o',s=30)
plt.show()


#21 3D Vector field with Matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.4))
u = np.sin(x) * np.cos(y) * np.cos(z)
v = -np.cos(x) * np.sin(y) * np.cos(z)
w = (np.sqrt(2.0 / 3.0) * np.cos( x) * np.cos(y) * np.sin( z))
fig=plt.figure()
ax=Axes3D(fig)
ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
plt.show()