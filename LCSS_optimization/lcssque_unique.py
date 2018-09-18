# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:59:35 2018

@author: zhangyaxu
"""

import numpy as npy

def find_lcsseque(s1, s2): 
    '''
     若 字符串 s1  s2 对换位置，获得的子序列可能会不同,
     需要返回各字符所在的位置
    '''
	 # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 
    # d用来记录转移方向
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 
     
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
        #print(s1[p1-1])
        #print(s2[p2-1])
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
    return ''.join(s),s1_loc ,s2_loc

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
    return ''.join(rst_que),rst_loc1,rst_loc2

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
    #print('\n s1: ',s1,'\t s2: ',s2,'\n')
    #print(que1,'\n',s1_loc1,'\n',s2_loc1)
    #print(que2,'\n',s1_loc2,'\n',s2_loc2)
    
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
    return que,s1_loc,s2_loc



if __name__=='__main__':
    s1='zahbzadfebg'
    s2='bzhadebg'
    que,s1_loc,s2_loc=unique_lcssque(s1,s2,keep_continue=True,mode='any')
    print(que,'\n',s1_loc,'\n',s2_loc)

