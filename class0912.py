#%% 1. Numbers
 
int_number = 1
float_number = 1.0
complex_number = 1 + 2j
round_float = round(1234.5678, 2)
str2int = int('213')
 
print(round_float)


#%%

import numpy as np

a=np.array([1,2,3])
print(a)

a_t=a[np.newaxis]
print(a_t)

b_t=a[:,np.newaxis]


list1=[]
for i in a_t:
    for j in i:
        list1.append(j)
        
print(list1)

print(b_t,b_t.shape)


# %%

s=np.array([[1,2,3],[4,5,6]])

print(s.max())
print(s.mean())

# %%

ma=np.array([1,2,3])
mb=np.array([4,5,6])
mc=np.hstack((ma,mb))
md=np.vstack((ma,mb))

print(mc)

print(md[:,:2])

tot=0
for i in md[:,:2]:
    for j in i:
        tot+=j
        
print(tot)
        


# %%

k= [ i for i in range(5)]
print(k)

# %%  物件函式 基礎

class test1:
    def __init__(self,a,b,c):
        self.a=a
        self.b=b
        self.c=c
        
    def bool_id(self):
        if self.c=="Vini Vici":
            return True
        else:
            return False
        
    def add(self,num1,num2):
        return num1+num2
        
    
answer=test1("EDM","W&W","Vini Vici")

print(answer.bool_id(),"/",answer.a)

print(answer.add(99,100))



# %%

class test2:
    def __init__(self,p1,p2,p3,p4):
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.p4=p4
       
    def list1(self):
        return [self.p1,self.p2,self.p3,self.p4]

oct=test2("john","jacob","mary","david")

list1=oct.list1()

for i in range(len(list1)):
    print("Hello"+" "+list1[i])
    
    



# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()            

# %%

import plotly.express as px
import plotly.graph_objs as go

from IPython.display import HTML
import pandas as pd


df_data=px.data.iris()

fig=px.histogram(df_data,x=df_data.columns[1],color="species")
fig.show()

# %%
fig.write_html("./demo.html")

# %%
colu=[df_data.columns[i] for i in range(len(df_data.columns)-2)]
fig2=px.scatter_matrix(df_data,dimensions=colu,color=df_data.columns[4])

fig2.show()
fig2.write_html("./demo2.html")
fig2.write_image("./demo.png")


#%%

fig3=px.scatter_3d(df_data,x=df_data.columns[0],y=df_data.columns[2],z=df_data.columns[3],color=df_data.columns[4],size=df_data.columns[1])
fig3.write_html("./test1.html")

# %% 複合式視覺化技巧   ( 使用marginal_y & marginal_x 函數 )

fig4=px.scatter(df_data,x="sepal_width",y="sepal_length",color="species",
                marginal_x="rug",marginal_y="box")

fig4.write_html("VScomplication.html")

# %%

list1=[0,1,2]
print(list1[-2])

# %%

print(26**4)

# %%
print(24/10000)
# %%

from scipy.special import comb
a1=comb(13,5)
a2=comb(13,4)
a3=comb(13,2)
a4=comb(13,2)
b1=comb(52,13)
print(a1*a2*a3*a4/b1)
print(round(a1*a2*a3*a4/b1,5))

# %%
print(4885/5000)



# %%
import sys
name=sys.stdin.readline()
print(name)

#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
x=np.linspace(-5,5,100)
f=(lambda x : x**2+3*x+5)

plt.title("$f(x)=x^2+3x+5$")
plt.plot(x,f(x))

sns.lineplot(x=x,y=f(x))

# %%
x2=np.arange(0,7,0.01)
f2=lambda x: x**3-10*x**2+29*x-20
plt.title("$f(x)=x^3-10x^2+29x-20$")
plt.grid(visible="True",linewidth=0.5)
sns.lineplot(x=x2,y=f2(x2),linestyle="--",color="g")
sns.despine()
plt.show()



# %%

x3=np.linspace(-5,10,100)
cof=[1,-8,16,-2,8]

# cof=多項式係數 ， x=帶入的x值
f3=lambda cof,x3 : np.polyval(cof,x3)

ax=plt.gca()
sns.lineplot(x=x3,y=f3(cof,x3),linestyle="--",color="g")
ax.grid(True)
ax.set_xlabel('x'),ax.set_ylabel("f(x)")
ax.set_title("The range of $x$ is not appropriate")
plt.savefig("poly.eps")

# %%
import numpy as np
print(np.arange(20))

for i in np.arange(20):
    print(i)
    

# %%
