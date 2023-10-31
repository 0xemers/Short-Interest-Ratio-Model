#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

SH_IN=pd.read_excel(r"E:\HKUST_CLASS\22spring\Investment_Model\project1_simplefactor\Data_origin.xlsx",sheet_name="SPX_Short Int Ratio")
CAP=pd.read_excel(r"E:\HKUST_CLASS\22spring\Investment_Model\project1_simplefactor\Data_origin.xlsx",sheet_name="SPX_Mkt Cap")
P=pd.read_excel(r"E:\HKUST_CLASS\22spring\Investment_Model\project1_simplefactor\Data_origin.xlsx",sheet_name="SPX_Last Price")
GICS=pd.read_excel(r"E:\HKUST_CLASS\22spring\Investment_Model\project1_simplefactor\Data_origin.xlsx",sheet_name="SPX_GICS")

# 3 month avg SH_IN_RATIO
SH_IN=SH_IN.set_index(["DATES"])
SH_IN.columns=pd.Index(SH_IN.columns,name="STOCK_NAME")

# 删除数据不满三个月的股票
SH_IN=SH_IN.sort_index()
delete=SH_IN.count()[SH_IN.count()<3].to_frame()
SH_IN=SH_IN.drop(delete.index.values,axis=1)

# calculation
avg_ratio= SH_IN.copy(deep=True)
avg_ratio=avg_ratio.fillna(0)
for j in range(len(avg_ratio.index)):
        if j<=1:
            avg_ratio.iloc[j]=0
        else:
            avg_ratio.iloc[j]=SH_IN.iloc[j-2:j+1].mean()
avg_ratio.drop(avg_ratio.head(2).index,inplace=True)

CAP=CAP.set_index(["DATES"])
CAP.columns=pd.Index(CAP.columns,name="STOCK_NAME")
CAP=CAP.drop(delete.index.values,axis=1) #把SH_IN缺失的股票删了

# Calculate WGT according to the Market CAP
CAP=CAP.fillna(0)
WGT= CAP.copy(deep=True)
for i in range(len(WGT.index)):
    WGT.iloc[i]= WGT.iloc[i]/sum(WGT.iloc[i])
WGT.head()


P=P.set_index(["DATES"])
P.columns=pd.Index(P.columns,name="STOCK_NAME")
P=P.drop(delete.index.values,axis=1) #把SH_IN缺失的股票删了
P=P.sort_index()
P=P.fillna(0)

# Calulate sub-1-mon-return
Return= P.copy(deep=True)
for i in range(len(Return.index)-1):
    Return.iloc[i]= (Return.iloc[i+1]-Return.iloc[i])/Return.iloc[i]
Return=Return.drop(Return.iloc[-1].name)
Return=Return.replace(np.inf,0)
Return.head()

# stack WGT RETURN SH_IN_RATIO
AVG_RATIO=avg_ratio.stack().to_frame().reset_index()
WEIGHT=WGT.stack().to_frame().reset_index()
SUBS_1M_USD=Return.stack().to_frame().reset_index()

data0=pd.merge(WEIGHT,SUBS_1M_USD, on=["DATES", "STOCK_NAME"], how='inner')
data1=pd.merge(data0,AVG_RATIO, on=["DATES", "STOCK_NAME"], how='inner')

# Merge GICS
GICS=GICS.set_index(["STOCK_NAME"])
data = pd.merge(data1, GICS, on="STOCK_NAME", how='left')
data.columns=["DT","STOCK_NAME","WEIGHT","SUBS_1M_USD","3M_AVG_RATIO","GICS"]
data["DT"]=pd.to_datetime(data["DT"]).dt.date
data.head()


# Save as excel
outputpath=r"E:\HKUST_CLASS\22spring\Investment_Model\project1_simplefactor\DATA.xlsx"
data.to_excel(outputpath,index="True",header="True")

