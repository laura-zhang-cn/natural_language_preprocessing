# -*- coding:utf-8 -*-
"""
Created on Mon Apr 12 11:15:46 2021

@author: yxzha

之前写的动态规划的lcss求解过程有点晦涩，重新写一版容易解释和理解的
这一版的算法复杂度与第一版的相同都是O(m×n)，其主要优点在于：
1.这一版算法更加易于解释和理解
2.不受字符串s1,s2参数传入顺序的影响
3.可以输出存在的多个可行解（第一版基于动态规划的算法即使对换两者位置，最多也只可能输出2个不同的最长公共子序列）

v_opt2021版本的核心思想：
将两个字符的匹配呈现到矩阵中（s1是行坐标，s2是类坐标），
每次循环都选择当前坐标块的子区域中包含最多匹配数的子坐标块，
直到当前坐标块不再存在子区域

"""

import pandas as pds
import numpy as npy

def lcss_extract(s1,s2):
    #s1='aded';s2='dd'
    #1 忽略顺序，先获得所有公共字符的位置对应坐标集合 mlst
    m0=pds.DataFrame(npy.zeros(shape=(len(s1),len(s2))))
    mlst=[]
    for idx1 in range(len(s1)) :
        for idx2 in range(len(s2)) :
            if s2[idx2]==s1[idx1]:
                m0.loc[idx1,idx2]=1.0
                mlst.append([idx1,idx2])
    # 
    rst=[] # 保留匹配的最长公共子序列合集，也是最终输出的结果集合
    if len(mlst)==0:
        return rst
    else:
        cols=['s1','s2']
        midx=pds.DataFrame(mlst,columns=cols)
        midx['margin']=midx[cols].apply(lambda x :npy.min(midx[cols]>x,axis=1).sum(),axis=1) # 右下子区域的匹配数，很重要
        midx=midx.sort_values(by=['margin'],ascending=[0]).reset_index(drop=True) # 一定要排序，确保更快找到最长公共子序列
        #2 初始化
        midx_tem=midx.copy() 
        lgst=0 # 记录获得的子序列的长度 : 会不停更新，直到：子区域匹配数没有超过这个长度的（即不可能再出现比这个更长的公共子序列）
        margins_id=[]  #已经使用过的区域合集，即margins_idx的集合，因为最长公共子序列不唯一，我们需要尽量找出不同顺序下的多个可行解（并非全部）
        rst_id=[] # 已经查找过的公共子序列的起始点index集合，即margins_idx的第一个值，避免重复查找
        #3 开始查找最长公共子序列：输出多个可行解并存储到rst中
        while True:
            midx_tem['sub_area']=True #初始化全域即为第一个子域
            rst_char='' #保留一次的最长公共子序列
            rst_lcx=[] # 保留一次的最长公共子序列在字符串中对应的索引位置
            margins_idx=[] # 保留一次查找时已经使用过的区域
            #4 保留当前区域的下层子域sub_area包含最多匹配数margin的字符位置(s1,s2)的索引值index
            while midx_tem.sub_area.sum()>0:
                #如果下层子域存在，开始查找子序列
                sub_area_max_margin=midx_tem.loc[midx_tem.sub_area,'margin']==midx_tem.loc[midx_tem.sub_area,'margin'].max()
                tem_margins_id=min(sub_area_max_margin[sub_area_max_margin].index)
                tem_rst_lcx=midx_tem.loc[tem_margins_id,cols].values
                rst_char=rst_char+s1[tem_rst_lcx[0]] 
                rst_lcx.append(list(tem_rst_lcx)) 
                margins_idx.append(tem_margins_id)
                midx_tem['sub_area']=npy.min(midx_tem[cols]>tem_rst_lcx,axis=1).values
            
            #5 更新最长公共子序列可行解集合rst
            if len(rst_lcx)==lgst:
                rst.append(npy.array(rst_lcx).T.tolist()+[rst_char])
            elif len(rst_lcx)>lgst:
                lgst=len(rst_lcx)
                rst=[npy.array(rst_lcx).T.tolist()+[rst_char]] # 获得新的更长序列，则直接替换所有rst旧的序列,因为里面的序列都是更短的
            else:
                pass # 不更新rst
            #6 更新已经计算过的区域的index ,并检查计算是否还可能存在可行解 让最外层循环继续
            rst_id.append(margins_idx[0])
            margins_id.extend(margins_idx) 
            other_margins_id=npy.setdiff1d(midx.index.values,npy.array(margins_id))
            if len(other_margins_id)>0 and sum(midx.loc[other_margins_id,'margin']>=lgst-1)>0:
                #7可能存在其它可行解，则需要更新数据，并继续最外层的循环
                midx_tem=midx.loc[npy.setdiff1d(midx.index.values,npy.array(rst_id)),:].copy()
                continue
            else:
                break
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


