## -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:52:16 2018

@author: Laura-Zhang

对流行的jaro算法提供了3种改进方案: 
improve1 
improve2 
improve3

'''
import math
def jaro(s1,s2,mode='winkler',h=3,p=0.1):
    '''
    s1,s2 : the text-string that will be calculate distince, length(s)>0
    mode :  ['popular' , 'improve1' , 'winkler' , 'improve1_winkler',......]
            popular  : use the most popular function,no `winkler`
            improve[1,2,3] : use 't' in another way to make the function more reasonable 
            winkler : jaro-winkler  ,an improvement of jaro popular method [default]
            improve[1,2,3]_winkler : both improve and winkler
    h : int, if mode is winkler,active h ,which means head-str number less then h
    p : float, if mode is winkler,active p , must control h*p<=1.0 ,otherwise will raise error
        if None,set p=0.5/h 
    '''
    # 计算 距离小于param1的有效匹配到的字符MatchValue
    ls1=len(s1);ls2=len(s2)
    param1=max(ls1,ls2)/2.0-1.0 # 有效匹配的条件 ：不超过最大容忍距离
    mval1=[];mval2=[] # match result.  
    ens1=list(enumerate(list(s1)))
    ens2=list(enumerate(list(s2)))
    for id1,val1 in ens1:
        for id2,val2 in ens2:
            if abs(id1-id2)<=param1 and val1==val2:
                mval1.append((id1,val1))
                mval2.append((id2,val2))
                # once matched not match again : both s1 and s2
                del ens2[ens2.index((id2,val2))] 
                break
    mval2.sort(key=lambda x:x[0]) # sort by original index in s2
    
    # 计算需要换位的次数 t
    mval=[(mval1[idx][1],mval2[idx][1]) for idx in range(len(mval1))]
    t=len([v1 for v1,v2 in mval if v1!=v2])/2.0
    
    # 计算 jaro-distinct  dj
    m=len(mval)*1.0
    if m==0 :
        alpha=0
        dj=0.0
    elif 'improve' not in mode:
        alpha=(m-t)/m
        dj=1.0/3*(m/ls1+m/ls2+alpha) # this function is popular 
    elif 'improve1' in mode:
        alpha=(m-t)/m
        dj=1.0/2*(m/ls1+m/ls2)*alpha  # use 't' in another way ,this is more reasonable
    elif 'improve2' in mode:
        alpha=(m-t)/m
        dj=(m/max(ls1,ls2))*alpha
    elif 'improve3' in mode:
        alpha=(m-t)/m
        w1=math.log(ls1+1,2)/(math.log(ls1+1,2)+math.log(ls2+1,2))
        w2=math.log(ls2+1,2)/(math.log(ls1+1,2)+math.log(ls2+1,2))
        dj=(w1*m/ls1+w2*m/ls2)*alpha  # not average_weight
    #print(t,m,alpha,dj)
    
    # mode包含winkler 时，触发接下来的计算
    if 'winkler' in mode:
        if h*p>1.0: print('h*p>1.0 ,this will cause distince>1')
        l=0
        h=min(len(s1[0:h]),len(s2[0:h]))
        for hx in range(h):
            if s1[hx]==s2[hx]:
                l=l+1.0
            else:
                break
        dw=dj+l*p*(1-dj)
        #print(l,dw)
        return dw
    else:
        return dj
        
if __name__=='__main__':
  s1=u'palladio'
  s2=u'palladium'
  
  s3=u'宾格binger'
  s4=u'宾格瑞bingerui'
  
  s5=u'binger'
  s6=u'bingerui'
  
  d12=jaro(s1,s2,mode='improve3_winkler',h=10,p=0.01)     #popular winkler improve1_winkler  improve2_winkler  ....
  d34=jaro(s3,s4,mode='improve3_winkler',h=3,p=0.1)    
  d56=jaro(s5,s6,mode='improve3_winkler',h=7,p=0.02)  
