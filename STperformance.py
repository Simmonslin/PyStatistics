#%%
from calendar import calendar
from email.charset import Charset
from os import remove
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 

import plotly.offline as py 
import plotly.tools as pls 
import plotly.figure_factory as ff
import cufflinks

#%%
import pandas as pd
st_perf=pd.read_csv("C:\Maths.csv")

import plotly.express as px
st_perf_school=st_perf["school"].value_counts()
px.pie(st_perf_school,values=st_perf_school.values,names=st_perf_school.index)


# %%
st_perf.groupby("Mjob")["Medu"].value_counts().plot(kind="count")

print(st_perf.groupby("Mjob")["Medu"].value_counts())

# %%


print(st_perf.pivot(columns="Mjob",values="Medu"))
print(st_perf.groupby("Mjob")["Medu"].value_counts()["health"])
# %%
import seaborn as sns
sns.countplot(st_perf,x="Mjob",hue="Medu")



# %%
print(st_perf.groupby("Mjob")["Medu"].mean())
# %%

sns.relplot(st_perf,x="studytime",y="traveltime",kind="line")
# %%


# %%

sns.pairplot(data=st_perf,hue="sex")

# %%

print(st_perf.groupby("sex")["romantic"].value_counts())
print(st_perf.groupby("activities")["romantic"].value_counts())
# %%
st_perf_acRO=st_perf[["sex","activities","romantic"]]

st_perf_acRO[["activities","romantic"]]=st_perf_acRO[["activities","romantic"]].applymap(lambda yes : 1 if yes=="yes" else 0)

# %%
import numpy as np

print(st_perf_acRO[["activities","romantic"]].corr())

# %%
sns.heatmap(st_perf_acRO[["activities","romantic"]].corr())
# %%
st_perf_acRO2=st_perf_acRO.copy()
st_perf_acRO2["sex"]=st_perf_acRO2["sex"].map({"M":0,"F":1})
sns.heatmap(st_perf_acRO2.corr(),annot=True)



# %%

