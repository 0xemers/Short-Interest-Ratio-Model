#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
data=pd.read_excel(r"E:\HKUST_CLASS\22spring\Investment_Model\project1_simplefactor\DATA.xlsx")
data=data.iloc[:,1:7]
data.head()
 
# Assign baskets
nobas=3
data['basket']=data.groupby('DT')['3M_AVG_RATIO'].rank(method='average',pct=True,ascending=True)
data['basket']=data['basket'].apply(lambda x : np.trunc((x-0.00000001)*nobas)+1)
 
# Return
def weighted_avg(x, weights):
    weights.iloc[x.isnull()==True]=0
    x=x.fillna(0)
    return sum(x*weights)/sum(weights)
perf=data.groupby(['DT','basket'])[['SUBS_1M_USD','WEIGHT']].apply(lambda x:weighted_avg(x['SUBS_1M_USD'], weights=x['WEIGHT'])).to_frame()
perf.columns=['perf_port']
 
# Excess Return
perf_bm=data.groupby(['DT'])[['SUBS_1M_USD','WEIGHT']].apply(lambda x:weighted_avg(x['SUBS_1M_USD'], weights=x['WEIGHT'])).to_frame()
perf_bm.columns=['perf_bm']
perf=perf.merge(perf_bm,how='right',left_index=True,right_index=True)
perf['perf_rel']=perf.perf_port-perf.perf_bm
 
perf=perf['perf_rel'].unstack(level = 1)
perf.columns= ['Low','Mid','High']
perf['LS']=perf.Low-perf.High
 
perf=perf.stack().to_frame()
perf.columns=['perf_rel']
ExRtn=perf.groupby(level=1).mean()*12
RiskAdjRtn=ExRtn/(perf.groupby(level=1).std()*12**0.5)
 
# Performance chart – 3 baskets
perf_cum=perf.groupby(level=1).cumsum()
perf_cum.unstack(level=1)['perf_rel'][['Low','Mid','High']].plot.line()
 
# Performance chart - Monthly and Cumulative Return
perf2 = data.groupby(['DT','basket'])[['SUBS_1M_USD','WEIGHT']].apply(lambda x: weighted_avg(x['SUBS_1M_USD'],weights=x['WEIGHT'])).to_frame()
perf2.columns = ['perf_port']
perf2 = perf2.unstack(level = 1)
perf2.columns = ['Low','Mid','High']
perf2['LS'] = perf2.Low - perf2.High
 
perf2 = perf2.stack().to_frame()
perf2.columns=['perf_port']
 
 
perf2_monthly=perf2.unstack(level=1)
perf2_monthly.columns=perf2_monthly.columns.droplevel()
perf2_monthly=perf2_monthly.reset_index()
 
perf2_cum = perf2.groupby(level=1).cumsum()
perf2_cum.columns=['perf_port_cum']
perf2_cum=perf2_cum.unstack(level=1)
perf2_cum.columns=perf2_cum.columns.droplevel()
perf2_cum=perf2_cum.reset_index()
 
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title('Monthly and Cumulative Return', fontsize=16)
ax1.set_xlabel('Year', fontsize=16)
ax1.set_ylabel('Monthly Return', fontsize=14)
ax1.bar(x=perf2_monthly.DT,height=perf2_monthly['LS'], width=20,label='Monthly Return')
ax1.set_xlim([datetime.strptime('2015-05-28','%Y-%m-%d'),datetime.strptime('2022-02-10','%Y-%m-%d')])
ax1.set_ylim([-0.04,0.065])
ax1.legend(loc='upper left')
ax2=ax1.twinx()
ax2.set_ylabel('Cumulative Return', fontsize=14)
ax2.plot(perf2_cum.DT,perf2_cum['LS'],'c',label='Cumulative Return',linewidth=5)
ax2.set_ylim([0,0.55])
ax2.legend(loc='upper right')
 
#Performance Contribution from Long and Short Positions
fig, ax1 = plt.subplots(figsize=(20, 6))
ax1.set_title('Performance Contribution from Long and Short Positions', fontsize=16)
ax1.set_xlabel('Year', fontsize=16)
ax1.set_ylabel('Monthly Return', fontsize=14)
ax1.bar(x=perf2_monthly.DT,height=perf2_monthly['Low'], width=20,label='Long')
ax1.bar(x=perf2_monthly.DT,height=-perf2_monthly['High'], width=20,label='Short')
ax1.set_xlim([datetime.strptime('2015-05-29','%Y-%m-%d'),datetime.strptime('2022-02-10','%Y-%m-%d')])
ax1.legend(loc='upper left')
 
 
# Information Coefficient
drank=data[['DT','SUBS_1M_USD','3M_AVG_RATIO']].copy()
drank[['3M_AVG_RATIO','SUBS_1M_USD']]=drank[['3M_AVG_RATIO','SUBS_1M_USD']].groupby(drank['DT']).rank()
ic=drank[['DT','3M_AVG_RATIO']].groupby('DT').corrwith(drank['SUBS_1M_USD'])
ic.plot.bar(y='3M_AVG_RATIO',title='Information Coefficient',color=np.where(ic['3M_AVG_RATIO'] > 0, 'forestgreen', 'crimson')).xaxis.set_major_locator(ticker.MultipleLocator(base=5))
 
# Hit rate
data=data.merge(perf_bm.reset_index(),how='left',left_on='DT',right_on='DT')
HR=data.groupby(['DT','basket'])[['SUBS_1M_USD','perf_bm']].apply(lambda x: sum(x.SUBS_1M_USD>x.perf_bm)/sum(x.SUBS_1M_USD.isnull()==False)).reset_index()
HR.columns=['DT','basket','Hitrate']
HR['Hitrate']=HR['Hitrate']-0.5
 
HR[HR.basket==1].plot.bar(x='DT',y='Hitrate', color=np.where(HR[HR.basket==1]['Hitrate'] > 0, 'forestgreen', 'crimson'),legend=None ,title='Long Leg - Lower Short Interest',figsize=(5,3),ylim=(-0.3,0.3)).xaxis.set_major_locator(ticker.MultipleLocator(base=5))
HR[HR.basket==3].plot.bar(x='DT',y='Hitrate', color=np.where(HR[HR.basket==3]['Hitrate'] > 0, 'forestgreen', 'crimson'),legend=None,title='Short Leg - Higher Short Interest',figsize=(5,3),ylim=(-0.3,0.3)).xaxis.set_major_locator(ticker.MultipleLocator(base=5))
 
#Turnover
#after rebalance
data['Basket_WEIGHT'] = data.groupby(['DT','basket'])['WEIGHT'].apply(lambda x: x/sum(x))
#before rebalance
data['Basket_WEIGHT1'] = [x*(1+y) for x,y in zip(data.Basket_WEIGHT,data.SUBS_1M_USD)]
data['Basket_WEIGHT1'] = data.groupby(['DT','basket'])['Basket_WEIGHT1'].apply(lambda x: x/sum(x))
 
#Short Leg
dat1 = data.loc[data.basket==3,].set_index(['DT','STOCK_NAME'])['Basket_WEIGHT'].unstack(level=1).fillna(0)
dat2 = data.loc[data.basket==3,].set_index(['DT','STOCK_NAME'])['Basket_WEIGHT1'].unstack(level=1).shift(1).fillna(0)
 
tcost=(dat1-dat2).stack().to_frame()
tcost.columns = ['turnover']
tcost.loc[tcost.index.get_level_values('DT')=='2015-05-29','turnover'] = 0
tcost['cost'] = [i*0.0033 if i>0 else -i*0.003432 for i in tcost.turnover]
turnover1 = tcost.groupby(level=0)['turnover'].apply(lambda x: sum(abs(x)))
print(f'Turnover of Short Leg: {turnover1.mean()}' )
 
perf = data[data.basket==3].groupby(['DT'])[['SUBS_1M_USD','Basket_WEIGHT']].apply(lambda x: weighted_avg(x['SUBS_1M_USD'],weights=x['Basket_WEIGHT'])).to_frame()
perf.columns = ['perf_precost']
tcost = tcost.groupby(level=0)['cost'].sum().to_frame()
perf = perf.merge(tcost, left_index = True, right_index = True)
perf['perf_aftercost'] = perf.perf_precost - perf.cost
print(f'\nPerformance of Short Leg before and after cost:\n {perf.mean()*12}\n')
 
#Long Leg
dat3 = data.loc[data.basket==1,].set_index(['DT','STOCK_NAME'])['Basket_WEIGHT'].unstack(level=1).fillna(0)
dat4 = data.loc[data.basket==1,].set_index(['DT','STOCK_NAME'])['Basket_WEIGHT1'].unstack(level=1).shift(1).fillna(0)
tcost2=(dat3-dat4).stack().to_frame()
tcost2.columns = ['turnover2']
tcost2.loc[tcost2.index.get_level_values('DT')=='2015-05-29','turnover2'] = 0
tcost2['cost2'] = [i*0.0033 if i>0 else -i*0.003432 for i in tcost2.turnover2]
turnover2 = tcost2.groupby(level=0)['turnover2'].apply(lambda x: sum(abs(x)))
print(f'Turnover of Long Leg: {turnover2.mean()}' )
 
perf = data[data.basket==1].groupby(['DT'])[['SUBS_1M_USD','Basket_WEIGHT']].apply(lambda x: weighted_avg(x['SUBS_1M_USD'],weights=x['Basket_WEIGHT'])).to_frame()
perf.columns = ['perf_precost']
tcost2 = tcost2.groupby(level=0)['cost2'].sum().to_frame()
perf = perf.merge(tcost2, left_index = True, right_index = True)
perf['perf_aftercost'] = perf.perf_precost - perf.cost2
print(f'\nPerformance of Long Leg before and after cost:\n {perf.mean()*12}')
 
 
 
# Sector tilt
gics_des=pd.read_csv(r"E:\HKUST_CLASS\22spring\Investment_Model\project1_simplefactor\GICS.csv")
sctwgt=data.groupby(['DT','basket','GICS'])['WEIGHT'].sum().reset_index()
sctwgt_scaler=data.groupby(['DT','basket'])['WEIGHT'].sum().reset_index()
sctwgt_scaler=sctwgt_scaler.rename(columns={'WEIGHT':'WEIGHT_scaler'})
sctwgt=sctwgt.merge(sctwgt_scaler,how='left',right_on=['DT','basket'],left_on=['DT','basket'])
sctwgt['WEIGHT']=sctwgt['WEIGHT']/sctwgt['WEIGHT_scaler']
sctwgt=sctwgt.drop(labels='WEIGHT_scaler',axis=1)
 
sctwgt_BM=data.groupby(['DT','GICS'])['WEIGHT'].sum().reset_index()
sctwgt_BM_scaler=data.groupby(['DT'])['WEIGHT'].sum().reset_index()
sctwgt_BM_scaler=sctwgt_BM_scaler.rename(columns={'WEIGHT':'WEIGHT_scaler'})
sctwgt_BM=sctwgt_BM.merge(sctwgt_BM_scaler,how='left',right_on='DT',left_on='DT')
sctwgt_BM['WEIGHT']=sctwgt_BM['WEIGHT']/sctwgt_BM['WEIGHT_scaler']
sctwgt_BM=sctwgt_BM.drop(labels='WEIGHT_scaler',axis=1)
sctwgt_BM=sctwgt_BM.rename(columns={'WEIGHT':'WEIGHT_BM'})
 
sctwgt=sctwgt.merge(sctwgt_BM,how='left',right_on=['DT','GICS'],left_on=['DT','GICS'])
sctwgt['WEIGHT_rel']=sctwgt['WEIGHT']-sctwgt['WEIGHT_BM']
gics_des['GICS1']=[str(i) for i in gics_des.GICS1]
gics_des['GICS1']=gics_des['GICS1'].apply(int) 
sctwgt=sctwgt.merge(gics_des,how='left',left_on='GICS',right_on='GICS1')
sctwgt_low=sctwgt[sctwgt.basket==1][['DT','Sector','WEIGHT_rel']].set_index(['DT','Sector']).unstack(level=1)
low_pl=sctwgt_low.WEIGHT_rel.plot.bar(stacked=True,figsize=(15,3),title = 'Long Leg - Lower Short Interest')
low_pl.legend(loc='center right',bbox_to_anchor=(1.23,0.5))
low_pl.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
sctwgt_high=sctwgt[sctwgt.basket==3][['DT','Sector','WEIGHT_rel']].set_index(['DT','Sector']).unstack(level=1)
high_pl=sctwgt_high.WEIGHT_rel.plot.bar(stacked=True,figsize=(15,3),title = 'Short Leg - Higher Short Interest')
high_pl.legend(loc='center right',bbox_to_anchor=(1.23,0.5))
high_pl.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
 
# sector-neutral
data['basket_s']=data.groupby(['DT','GICS'])['3M_AVG_RATIO'].rank(method='average',pct=True,ascending=True)
data['basket_s']=data['basket_s'].apply(lambda x : np.trunc((x-0.00000001)*nobas)+1)
perf_s=data.groupby(['DT','GICS','basket_s'])[['SUBS_1M_USD','WEIGHT']].apply(lambda x:weighted_avg(x['SUBS_1M_USD'], weights=x['WEIGHT'])).to_frame()
perf_s.columns=['perf_port_s']
 
perf_s=perf_s.reset_index().merge(sctwgt_BM,how='left',right_on=['DT','GICS'],left_on=['DT','GICS']).groupby(['DT','basket_s'])[['perf_port_s','WEIGHT_BM']].apply(lambda x:sum(x.perf_port_s*x.WEIGHT_BM)/sum(x.WEIGHT_BM)).to_frame()
perf_s.columns=['perf_port_sctneut']
 
perf_s=perf_s.merge(perf_bm,how='right',left_index=True,right_index=True)
perf_s['perf_rel_sctneut']=perf_s.perf_port_sctneut-perf_s.perf_bm
perf_s=perf_s['perf_rel_sctneut'].unstack(level=1)
perf_s.columns=['Low','Mid','High']
perf_s['LS']=perf_s.Low-perf_s.High
perf_s=perf_s.stack().to_frame()
perf_s.columns=['perf_rel_sctneut']
ExRtn_s=perf_s.groupby(level=1).mean()*12
RiskAdjRtn_S=ExRtn_s/(perf_s.groupby(level=1).std()*12**0.5)
 
perf_s_cum=perf_s.groupby(level=1).cumsum()
a=perf_cum.merge(perf_s_cum,right_index=True,left_index=True)
a.columns = ['Across Market','Sector Neutral']
plot1=a.iloc[a.index.get_level_values(1)=='LS'].plot.line(figsize=(10,5),linewidth=5)
plot1.set_xlabel('Date')
plot1.set_ylabel('Cumulative Return')
plot1.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
plot1.set_xticklabels(plot1.get_xticklabels(), rotation=45)

