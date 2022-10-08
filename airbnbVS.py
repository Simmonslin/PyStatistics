
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

import chardet
with open ("/Users/ASUS/Documents/AirbnbListing.csv","rb") as r:
    result=chardet.detect(r.read())
    
print(result)


# %%

import pandas as pd

airbnb_data=pd.read_csv("/listings.csv")



# %%
print(airbnb_data.columns)


# %%
import seaborn as sns
sns.barplot(data=airbnb_data,
            x=airbnb_data.has_availability.value_counts().index,
            y=airbnb_data.has_availability.value_counts().values,
            )

# %%

calendar_data=pd.read_csv("/calendar.csv")

# %%

print(calendar_data["available"].value_counts())
sns.barplot(data=calendar_data,
            x=calendar_data.available.value_counts().index,
            y=calendar_data.available.value_counts().values)

# %%
new_calendar=calendar_data[["date","available"]]
new_calendar["busy"]=new_calendar["available"].map(lambda x: 0 if x=="t" else 1)
new_calendar=new_calendar.groupby("date")["busy"].mean().reset_index()
new_calendar["date"]=pd.to_datetime(new_calendar["date"])


# %%
import matplotlib.pyplot as plt

plt.plot(new_calendar["date"],new_calendar["busy"])
plt.show()


#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(data=new_calendar,x="date",y="busy")
plt.title("Taipei Calendar")


# %% strftime 取時間序列中的特定值 ， e.g: January, February


price_calendar=calendar_data[["date","price"]]
price_calendar["month"]=pd.to_datetime(price_calendar["date"]).dt.strftime("%B")
price_calendar["price"]=price_calendar["price"].str.replace(",","").str.replace("$","").astype(float)

price_calendar=price_calendar.groupby(("month"),sort=False)["price"].mean().reset_index()

# %%
price_calendar_num=calendar_data[["date","price"]]
price_calendar_num["month"]=pd.to_datetime(price_calendar_num["date"]).dt.month.astype(str).map(lambda x:(str(x)+"月"))
price_calendar_num["price"]=price_calendar_num["price"].str.replace(",","").str.replace("$","").astype(float)
price_calendar_num=price_calendar_num.groupby("month")["price"].mean().reset_index()

# %%
from matplotlib import rcParams
rcParams['font.family'] = 'SimSun'
#%%
sns.barplot(data=price_calendar_num,x="price",y="month")

# %%
price_calendar_num["price"]=sorted(price_calendar_num["price"],reverse=True)

# %%

calendar_data["week"]=pd.to_datetime(calendar_data["date"]).dt.day_name()
calendar_data["price"]=calendar_data["price"].str.replace(",","").str.replace("$","").astype(float)

#%%
price_week=calendar_data.groupby("week")["price"].mean().reset_index()
price_week["week"]=calendar_data["week"].unique()

sns.lineplot(data=price_week,x="week",y="price",color="orange")


# %% 寫法 1 : 直接用value_counts
district_count=pd.Series(airbnb_data.neighbourhood_cleansed.value_counts())

sns.barplot(x=district_count.index,y=district_count.values,color="green")
# %% 寫法2 : 先用groupby將不同區分隔，然後再從分類中挑出id項計算

print(airbnb_data.groupby("neighbourhood_cleansed").count()["id"])

# %%

sns.distplot(airbnb_data.review_scores_rating.dropna(),rug=True)
sns.distplot(airbnb_data.price.str.replace(",","").str.replace("$","").astype(float).dropna(),rug=True)

# 移除上方+右方邊線
sns.despine()

# %%
print(airbnb_data.price.str.replace(",","").str.replace("$","").astype(float).describe())

# %%
print(min(airbnb_data.price.str.replace(",","").str.replace("$","").astype(float)))
# %%
airbnb_data["price"]=airbnb_data["price"].str.replace(",","").str.replace("$","").astype(float)

#%%

Airbnb_price_extre=airbnb_data[airbnb_data["price"]==sorted(airbnb_data["price"],reverse=False)[1]]
Airbnb_price_extre=pd.concat([Airbnb_price_extre,airbnb_data[airbnb_data["price"]==sorted(airbnb_data["price"],reverse=True)[0]]])
# %%
def get_iloc():
    sub=0
    for i in Airbnb_price_extre.columns:
        sub+=1
        if i=="description":
            return sub-1
        else:
            continue
        
print(Airbnb_price_extre.iloc[0,get_iloc()])

#%%

sns.distplot(airbnb_data.price[airbnb_data["price"]<50000].dropna(),rug=True)
sns.despine()
# %%
fig=px.histogram(airbnb_data.price[(airbnb_data["price"]<=10000)&(airbnb_data["price"]>300)],title="Room Price Distribution")
fig.update_layout(title_font_size=30)
fig.write_html("price.html")


# %%

drop_outlier_price=airbnb_data[(airbnb_data["price"]<10000)&(airbnb_data["price"]>300)]
# %%
price_order=drop_outlier_price.groupby("neighbourhood_cleansed")["price"].median().sort_values(ascending=False)
sns.boxplot(data=drop_outlier_price,x="neighbourhood_cleansed",y="price",order=price_order.index)
# %%

def boxplot_to_price(column_name):
    drop_outlier_price=airbnb_data[(airbnb_data["price"]<10000)&(airbnb_data["price"]>300)]
    price_order=drop_outlier_price.groupby(column_name)["price"].median().sort_values(ascending=False)
    return sns.boxplot(data=drop_outlier_price,x=column_name,y="price",order=price_order.index)

boxplot_to_price("room_type")




# %%

print(airbnb_data["property_type"].unique())

# %%

fig_property=px.box(drop_outlier_price,x="property_type",y="price",category_orders=price_order.index)
fig_property.write_html("property.html")

#%%

drop_outlier_price.pivot(columns="property_type",values="price").iplot(kind="box")

# %%

airbnb_data.amenities=airbnb_data.amenities.str.replace('"',"").str.replace('[{}]',"")

# %%
all_item=np.concatenate(airbnb_data.amenities.map(lambda m: m.split(',')))

top_20_item=pd.Series(all_item).value_counts().head(20)


top_20_item.plot(kind="bar",color="yellow")


#%%
ame_items=airbnb_data["amenities"].str.replace('"',"").str.replace('[[]{}]',"")

ame_items=pd.Series(np.concatenate(ame_items.map(lambda m : m.split(",")))).value_counts().head(20)


#%%
print(top_20_item.index)

# %%

print(drop_outlier_price.pivot(columns="room_type",values="price").plot(kind="hist"))
# %%

print(drop_outlier_price.columns)

# %%

amenities=np.unique(np.concatenate(airbnb_data["amenities"].map(lambda ams: ams.split(","))))
amenity_price=[(amn,airbnb_data[airbnb_data["amenities"].map(lambda amns : amn in amns)].price.mean()) for amn in amenities if amn!=""]
amenity_series=pd.Series(data=[data[1] for data in amenity_price],index=[data[0] for data in amenity_price])


# %%


print([airbnb_data[airbnb_data["amenities"].map(lambda amns : amn in amns)].price for amn in amenities if amn!=""])


# %%

print(np.unique(np.concatenate(airbnb_data["amenities"].map(lambda ams: ams.split(",")))))

# %%

amenity_series.sort_values(ascending=False)[:20].plot(kind="bar")

# .gca()設定座標
ax=plt.gca()

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha="right",fontsize=12)
plt.show()

# %%

# bin= 直條圖數量 

print(drop_outlier_price.pivot(columns="beds",values="price").plot(kind="hist",stacked=True,bins=100))

# %%
drop_outlier_price["reviews_per_month"]=drop_outlier_price["reviews_per_month"].map(lambda num: round(num,1))

drop_outlier_price.pivot(columns="reviews_per_month",values="price").plot(kind="hist",stacked=True)

# %%

