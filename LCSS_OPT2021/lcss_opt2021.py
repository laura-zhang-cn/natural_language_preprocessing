# -*- coding:utf-8 -*-
"""
Created on Mon Apr 12 11:15:46 2021

@author: yxzha

之前写的动态规划的lcss求解过程有点晦涩，重新写一版容易解释和理解的
这一版的算法复杂度是O(m^2/d)，其主要优点在于：
1.这一版算法更加易于解释和理解
2.不受字符串s1,s2参数传入顺序的影响
3.可以输出所有存在的可行解（第一版基于动态规划的算法即使对换两者位置，最多也只可能输出2个不同的最长公共子序列）

v_opt2021版本的核心思想：
将两个字符的匹配呈现到矩阵中（s1是行坐标，s2是类坐标），
每次循环都选择当前坐标块的子区域中包含最大匹配深度的子坐标块，
直到当前坐标块不再存在子区域

"""

import pandas as pds
import numpy as npy

def lcss_extract(s1,s2):
    #共现坐标提取
    m0=pds.DataFrame(npy.zeros(shape=(len(s1),len(s2))))
    mlst=[]
    for idx1 in range(len(s1)) :
        for idx2 in range(len(s2)) :
            if s2[idx2]==s1[idx1]:
                m0.loc[idx1,idx2]=1.0
                mlst.append([idx1,idx2])
   #开始计算深度
    if len(mlst)==0:
       pass
    else:
        cols=['s1','s2']
        midx=pds.DataFrame(mlst,columns=cols)
        midy=midx.copy()
        midx['deep']=0 # 初始化深度为0
        midy['margin']=midy[cols].apply(lambda x : npy.sum(npy.min(midy[cols]>x,axis=1)),axis=1) # 右下子区域的匹配数，很重要
        while len(midy)>0:
            #zerodeep=midy.loc[midy.deep==0,cols]
            midy=midy.loc[midy.margin>0,cols]
            midx.loc[midy.index,'deep']=midx.loc[midy.index,'deep']+1 #所有存在子域的匹配点深度+1
            midy['margin']=midy[cols].apply(lambda x : npy.sum(npy.min(midy[cols]>x,axis=1)),axis=1)
        subdeeps=sorted(set(midx.deep),reverse=True)
        rst_loc=[]
        rst_char=[]
        st=1
        for dx in subdeeps:
            for i1,i2 in midx.loc[midx.deep==dx,cols].values:
                if  st:
                    rst_loc.append([(i1,i2)])
                    rst_char.append([s1[i1]])
                else:
                    for j,locs in enumerate(rst_loc):
                        j1,j2=locs[-1]
                        if j1<i1 and j2<i2:
                            rst_loc[j]=rst_loc[j]+[(i1,i2)]
                            rst_char[j]=rst_char[j]+[s1[i1]]
            st=0
        #整合输出
        rst_char=[''.join(x) for x in rst_char]       
        rst_loc=[npy.array(y).T.tolist() for y in rst_loc]
        rst=list(zip(rst_char,rst_loc))
        return rst

    
                
                

if __name__=='__main__':
    test_s=[['abca','baca'],['degag','gwgdfad'],['abchigdef','defhigabc'],
            [u'米庙镇尚庄乡农村淘宝昌源202西柚','米庙镇尚庄乡农村淘宝早餐店102葡萄'],
            [u'龙溪路63号附近艺海天成公寓A单元','下关龙溪路艺海天成公寓A单元']]
    '''
    s1,s2=[u'龙溪路63号附近艺海天成公寓A单元','下关龙溪路艺海天成公寓A单元']
    rst=lcss_extract(s1=s1,s2=s2)
    print(pds.DataFrame(rst,columns=[s1,s2,'lcss']))
    '''
    for s1,s2 in test_s:
        rst=lcss_extract(s1=s1,s2=s2)
        print(pds.DataFrame(rst,columns=[s1,s2,'lcss']))


