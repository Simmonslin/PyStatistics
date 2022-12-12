import numpy as np 
import matplotlib.pyplot as plt 

f_banana= lambda x,y : 100*np.square(y-np.square(x))+np.square(1-x)

x=np.linspace(-1.5,1.5,10)
y=np.linspace(-1.5,1.5,10)
X_b,Y_b=np.meshgrid(x,y)
Z_b=f_banana(X_b,Y_b)

fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X_b, Y_b, Z_b, color ='blue',
    alpha=0.3, rstride = 1, cstride = 1) # rstride : x 密度 ， cstride : y 密度
ax.set_xlabel('X'), ax.set_ylabel('Y')
ax.set_zlabel('f(X,Y)')
ax.view_init(10, -60)  #(elev=-165, azim=60)
plt.title('Wireframe (Mesh) Plot')
plt.show()