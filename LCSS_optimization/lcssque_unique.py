# -*- coding: utf-8 -*-
"""
Created on Fri Nov 31 10:59:35 2018

@author: zhangyaxu


lcss 
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

# from pyspark.sql import functions as F
# from pyspark.storagelevel import StorageLevel
#
# import pandas as pds
import math
import numpy as npy
import  re


def preprocess_str(s):
    '''
    一般情况下，字符串可以直接索引获取每个字符，而预处理字符串的目的在于判断不同类型的字符是否每个字符都是独立，
    这里，把连续的数值作为一个整体，把连续的英文也作为一个整体，只有中文将每个字符独立使用,注意最终不包含标点符号。
    :param s: a string
    :return: list of each str after processing
    '''
    s=s+'.'
    rst=[]
    num_pat=re.compile('[0-9]')
    eng_pat=re.compile('[a-zA-Z]')
    cn_pat=re.compile(u'[\u4e00-\u9fa5]')
    next_continue='cn'
    for ii in range(len(s)):
        condn=num_pat.match(s[ii])
        conde=eng_pat.match(s[ii])
        condc=cn_pat.match(s[ii])
        if condc:
            if next_continue!='cn':
                rst.append(tem)
                tem=''
            else:
                tem=''
            next_continue='cn'
            tem=tem+s[ii]
            rst.append(tem)
        elif condn:
            if next_continue=='eng':
                rst.append(tem)
            if next_continue!='num':
                tem=''
            tem=tem+s[ii]
            next_continue = 'num'
        elif conde:
            if next_continue=='num':
                rst.append(tem)
            if next_continue!='eng':
                tem=''
            tem=tem+s[ii]
            next_continue='eng'
        else:
            if next_continue=='num' or next_continue=='eng':
                rst.append(tem)
                tem=''
            next_continue='cn'
    return rst


def find_lcsseque(s1, s2):
    '''
     若 字符串 s1  s2 对换位置，获得的子序列可能会不同,
     需要返回各字符所在的位置
    '''
	# 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 
    # d用来记录转移方向
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]
    #
    for p1 in range(len(s1)): 
        for p2 in range(len(s2)): 
            if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 'left_up'          
            elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2] 
                d[p1+1][p2+1] = 'left'          
            else:                           #上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]   
                d[p1+1][p2+1] = 'up'         
    (p1, p2) = (len(s1), len(s2)) 
    #print numpy.array(d)
    s = [] 
    s1_loc=[]
    s2_loc=[]
    while m[p1][p2]:    #不为 0时
        c = d[p1][p2]
        # print(s1[p1-1])
        # print(s2[p2-1])
        if c == 'left_up':   #匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1-1])
            s1_loc.append(p1-1)  # 记录 匹配成功字符所在的位置
            s2_loc.append(p2-1)  # 记录 匹配成功字符所在的位置
            p1-=1
            p2-=1 
        if c =='left':  #根据标记，向左找下一个
            p2 -= 1
        if c == 'up':   #根据标记，向上找下一个
            p1 -= 1
    s.reverse() 
    s1_loc.reverse() 
    s2_loc.reverse() 
    return s,s1_loc ,s2_loc


def keep_continues_str(que,loc1,loc2,mode='any'):
    '''
    must:length que = length loc1 = length loc2 
    mode: in [both ,any ] ,means  continues-str-both or  continues-str-any  
    '''
    l=len(que)
    lx=1
    is_continue=0 # 判断 已判断的前一个字符 是否 与 更前一个字符是连续的，初始化为0不连续 ，连续则为1
    rst_que=[]
    rst_loc1=[]
    rst_loc2=[]
    while lx<l:
        #print(lx)
        if mode=='both':
            ldiff_left=max(loc1[lx]-loc1[lx-1],loc2[lx]-loc2[lx-1])
            #ldiff_right=max(loc1[lx+1]-loc1[lx],loc2[lx+1]-loc1[lx])
        if mode=='any':
            ldiff_left=min(loc1[lx]-loc1[lx-1],loc2[lx]-loc2[lx-1])
            #ldiff_right=min(loc1[lx+1]-loc1[lx],loc2[lx+1]-loc1[lx])
        if ldiff_left==1 and lx<l-1 :
            rst_que.append(que[lx-1])
            rst_loc1.append(loc1[lx-1])
            rst_loc2.append(loc2[lx-1])
            is_continue=1
        elif ldiff_left==1 and lx==l-1 :
            rst_que.extend([que[lx-1],que[lx]])
            rst_loc1.extend([loc1[lx-1],loc1[lx]])
            rst_loc2.extend([loc2[lx-1],loc2[lx]])
            is_continue=1
        elif ldiff_left>1 and is_continue==1: 
            rst_que.append(que[lx-1])
            rst_loc1.append(loc1[lx-1])
            rst_loc2.append(loc2[lx-1])
            is_continue=0
        lx=lx+1
    return rst_que,rst_loc1,rst_loc2


def keep_longest_continues(loc1,loc2):
    diff_loc1=npy.diff(npy.array(loc1)).mean()
    diff_loc2=npy.diff(npy.array(loc2)).mean()
    if diff_loc1<=diff_loc2:
        return (1,loc1)
    else:
        return (2,loc2)

def unique_lcssque(s1,s2,keep_continue=True,mode='any'):
    que1,s1_loc1,s2_loc1=find_lcsseque(s1, s2)
    que2,s2_loc2,s1_loc2=find_lcsseque(s2, s1)
    # print('\n s1: ',s1,'\t s2: ',s2,'\n')
    # print(que1,'\n',s1_loc1,'\n',s2_loc1)
    # print(que2,'\n',s1_loc2,'\n',s2_loc2)
    if keep_continue==True:
        que1,s1_loc1,s2_loc1=keep_continues_str(que=que1,loc1=s1_loc1,loc2=s2_loc1,mode=mode)
        que2,s1_loc2,s2_loc2=keep_continues_str(que=que2,loc1=s1_loc2,loc2=s2_loc2,mode=mode)
    #print(que1,'\n',s1_loc1,'\n',s2_loc1)
    #print(que2,'\n',s1_loc2,'\n',s2_loc2)
    if que1==que2:
        que=que1
        s1_loc=s1_loc1
        s2_loc=s2_loc1
    else:
        k,s_loc=keep_longest_continues(loc1=[s1_loc1,s2_loc1],loc2=[s1_loc2,s2_loc2])
        if k==1:
            que=que1
            s1_loc=s1_loc1
            s2_loc=s2_loc1
        else:
            que=que2
            s1_loc=s1_loc2
            s2_loc=s2_loc2
    return ''.join(que),s1_loc,s2_loc


def roundup(x,n):
    k=10**n
    return math.ceil(x*k)*1.0/k

def lcss_distance(ls1,ls2,s1_loc,s2_loc,credible_len=10):
    '''有序队列，靠前加权 ; '''
    ls1_w=sum(range(ls1+1))*1.0
    ls2_w=sum(range(ls2+1))*1.0
    s1_sim=sum((ls1-npy.array(s1_loc))/ls1_w)
    s2_sim=sum((ls2-npy.array(s2_loc))/ls2_w)
    # v1.0策略：基于两个相似度和的衰减策略 不可行，数据出来发现 一个相似度较低没关系，只要其中一个相似度足够大即可
    #decay=1.0 if s1_sim+s2_sim>=1.0 else s1_sim+s2_sim
    #
    # v1.1 策略： 通过跑出来一部分结果观察，发现文本越短，少量几个靠前的字符串相同，相似度更容易高于阈值，所以引入了基于文本长度的衰减策略来表示靠前加权距离的可信度
    # 阈值 是模拟了靠前加权结果在不同文本长度的情况下:
    # cumsum高于0.9时的靠前字符串的数量一般占总文本长度的比例，发现最小约为0.7，这里我们做了一点精度调节选为0.75;
    # cumsum高于0.7时的靠前字符串的数量一般占总文本长度的比例，发现最小约为0.5;
    len_rate_thr=[0.75,0.5]
    short_len_thr=roundup(min(ls1,ls2)*len_rate_thr[0],0)
    long_len_thr=roundup(max(ls1,ls2)*len_rate_thr[1],0)
    decay= 1.0 if len(s1_loc)*1.0>=short_len_thr and (len(s1_loc)*1.0>=long_len_thr or len(s1_loc)*1.0>=credible_len) else len(s1_loc)*1.0/max(short_len_thr,long_len_thr)
    #
    sim=float(max(s1_sim,s2_sim)*decay) #避免衰减后 ，取较大值的衰减结果比 二者中较小的 更小，所以 在衰减结果和较小值中选较大的那个
    print(s1_sim,'\n',s2_sim,'\n',sim)
    return  sim

def long_len_rate_thr_check():
    '''
    模拟：靠前加权法计算的初始距离：在不同文本长度情况下时，达到高相似的靠前长度占总长度的比例
    仅仅是 lcss_distance中decay所需的参数len_rate_thr的合适值的选择过程，不随整体算法运行。
    '''
    for ls1 in range(1,41,1):
        # ls1=6
        x=list(range(0,ls1+1))
        ls1_w=sum(x)
        y=(ls1-npy.array(x[0:ls1]))/ls1_w
        #print(y)
        z=npy.cumsum(y)
        zl=len(z[z>0.9])
        print('\t',ls1,'\t',len(z)-zl+1,'\t',(len(z)-zl+1)/len(z))

if __name__=='__main__':
    confx = SparkConf().setAppName('order location clustering')
    sc = SparkContext(conf=confx)
    sqlContext = HiveContext(sc)
    sc.setLogLevel("WARN")
    # 0 几个示例分别观察
    ## sim高的4个例子
    ss1=u'米庙镇尚庄乡农村淘宝昌源202西柚'
    ss2=u'米庙镇尚庄乡农村淘宝早餐店102葡萄'

    ss1=u'矿务局十字路口东豫华鞋城'
    ss2=u'矿务局豫华鞋城'

    ss1=u'川沙镇和平南路和平村陈家宅8号'
    ss2=u'川沙新镇和平南路陈家宅8号'

    ss1=u'徐丰公路东50米庞庄幼儿园(工人路)'
    ss2=u'徐丰公路(昕南社区卫生服务站西南)庞庄支点双语工幼儿园(工人路)'

    ss1=u'龙溪路63号附近艺海天成公寓A单元'
    ss2=u'下关龙溪路艺海天成公寓A单元'

    ## sim适中的2个例子
    ss1=u'大理大学荷花校区'
    ss2=u'大理大学下关校区'

    ss1=u'下关关平路86号'
    ss2=u'下关关平路下段迎运来超市'

    ## sim低的3个例子
    ss1 = u'庞庄支点双语幼儿园泉山区徐丰公路(昕南社区卫生服务站西南)庞庄支点双语工幼儿园(工人路)'
    ss2=u'徐丰公路东50米庞庄派出所'

    ss1=u'海东镇上河村蔚文街北附育英实验学校'
    ss2=u'海东镇向阳街'

    ss1=u'下关镇龙祥苑'
    ss2=u'下关镇大波罗甸21号'

    # 1 特殊预处理，仅中文情况下需要这样做，其它场景若不需要可直接到第2步：
    s1=preprocess_str(s=ss1) # 特殊处理：无标点符号，连续数字或连续字母均只占一个位置
    s2=preprocess_str(s=ss2) # 特殊处理：无标点符号，连续数字或连续字母均只占一个位置
    #
    # 2 LCSS
    que,s1_loc,s2_loc=unique_lcssque(s1,s2,keep_continue=True,mode='both')
    # 3 distince
    ls1=len(s1)
    ls2=len(s2)
    sim=lcss_distance(ls1, ls2, s1_loc, s2_loc,credible_len=10)
    #
    # 4 check result
    print('预处理后长度： \t',len(s1),'\t',len(s2))
    print('连续公共子序列： \n',que,'\n',s1_loc,'\n',s2_loc,'\n',len(s1_loc))
    print('lcss序列相似度： \t',sim)
