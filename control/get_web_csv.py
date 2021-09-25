#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def transform(x):
    if pd.isnull(x):
        x='unknow'
        return x;
    if x == 0:
        x='benign'
        return x
    if x>0:
        x='malware'
        return x

def get_count():
    df = pd.read_csv('latest.csv', usecols=['sha256','vt_detection'])
    new = [{'vt_class':'all','count':df.shape[0]}]
    df['vt_class'] = df['vt_detection'].apply(lambda x:transform(x))
    vt_class = df['vt_class'].groupby(df['vt_class']).count()
    vt_class = pd.DataFrame({'count':vt_class})
    vt_class.reset_index(inplace=True)
    vt_class = vt_class.append(new,ignore_index = True)
    print(vt_class)
    vt_class.to_csv('count.csv')

def get_time():
    df = pd.read_csv('latest.csv',usecols=['sha256','dex_date','vt_detection'])
    df.columns = ["sha256","date","vt_detection"]
    df['vt_class'] = df['vt_detection'].apply(lambda x:transform(x))
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    s = pd.Series(df['sha256'], index=df.index)
    count = s.resample('AS').count().to_period('A')# 按年统计并显示 
    result = pd.DataFrame({'all':count})
    result.reset_index(inplace=True)
    result["date"] = result["date"].astype("str")
    grouped = df.groupby(df['vt_class'])
    for name,group in grouped:
        print(name)
        print(group)
        s = pd.Series(group['sha256'], index=group.index)
        count = s.resample('AS').count().to_period('A')# 按年统计并显示 
        date_count = pd.DataFrame({name:count})
        date_count.reset_index(inplace=True)
        date_count["date"] = date_count["date"].astype("str")
        result = date_count.merge(result, how='left',on="date")
    result = result.loc[(result['date']>2010)&(result['date']<2021)]
    result.to_csv('time.csv')
    
def get_size():
    df = pd.read_csv('latest.csv',usecols=['sha256','apk_size','vt_detection'])
    listBins = list(range(0,101))
    listBins.append(10000)
    print(listBins)
    listLabels = list(range(1,102))
    print(listLabels)
    df["apk_size"]=df["apk_size"].apply(lambda x: x/1024/1024)
    df['apk_size'] = pd.cut(df['apk_size'], bins=listBins, labels=listLabels, include_lowest=True)
    count = df['apk_size'].groupby(df['apk_size']).count()
    print(11111)
    result = pd.DataFrame({'all':count})
    result.reset_index(inplace=True)
    df['vt_class'] = df['vt_detection'].apply(lambda x:transform(x))
    grouped = df.groupby(df['vt_class'])
    for name,group in grouped:
        print(name)
        print(group)
        apk_size = group['apk_size'].groupby(group['apk_size']).count()
        apk_size = pd.DataFrame({name:apk_size})
        apk_size.reset_index(inplace=True)
        result = apk_size.merge(result, how='left',on="apk_size")
    result.to_csv('size.csv')
    return result

def get_market():
    pd.set_option('display.max_columns', None)
    df = pd.read_csvi("latest.csv")
    market = df['dex_size'].groupby(df['markets']).count()
    market = pd.DataFrame({'count':market})
    market.reset_index(inplace=True)
    market['markets']=market['markets'].str.replace("play.google.com","google")
    market.to_csv('market.csv')
    return market

def get_vendors():
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("./data/json.csv")
    df["detected"]=df["detected"].astype("str")
    df1 = df[df.detected == "True"]
    vendor = df1['sha256'].groupby(df['company']).count()
    vendor = pd.DataFrame({'count':vendor})
    vendor.reset_index(inplace=True)
    vendor.to_csv('vendors.csv')
    return vendor

def get_positives():
    df = pd.read_csv("./data/jsoninfo.csv")
    positives = pd.DataFrame(df, columns=["sha256","positives"])
    positives = positives.sort_values(by="positives", ascending=False)
    positives.reset_index(drop=True, inplace=True)
    positives.to_csv('positives.csv')
    return positives    

def get_type():
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("./data/myfiletype.csv")
    filetype = df['sha256'].groupby(df['filetype']).count()
    filetype = pd.DataFrame({'count':filetype})
    filetype.reset_index(inplace=True)
    filetype['filetype']=filetype['filetype'].str.replace("Zip archive data  at least v2.0 to extract","Zip v2.0").str.replace(" Java archive data","")
    filetype.to_csv('type.csv')
    return filetype

def get_samples_number():
    df = pd.read_csv("./data/myfiletype.csv")
    print(df.shape[0])
    df1 = pd.read_csv("./data/jsoninfo.csv")
    print(df1.shape[0])
    number = {'samples':[df.shape[0]],'jsons':[df1.shape[0]]}
    number = pd.DataFrame(number)
    number.to_csv('number.csv')
    return df.shape[0],df1.shape[0]



if __name__ == '__main__':
    #get_count()
    get_time()
    #get_size()
    #get_market()
    #get_vendors()
    #get_positives()
    #get_samples_number()
    #get_type()

