# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:29:27 2018

@author: zhangyaxu

商品描述/标题等数据源 
训练词的word2vec向量并计算相似度，返回topn个，并和词的jaro距离 同时存储到hive中
"""


from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

import math
import gc
import pandas as pds
import numpy as npy

from gensim.models import Word2Vec
#from gensim.models import FastText

from params import  params

database_name=params['shared']['database_name']

def word2vec_most_sim(sentencs_split_list,size=100, window=5, min_count=1, workers=4,topn=10,sim_min=-1.0):
    '''
    通过word2vec向量，计算相似度，获得最相似的topn个结果，
    return pandas-dataframe which columns are ['target_word','sim_word','similarity']
    '''
    #reload(sys) ;    sys.setdefaultencoding('utf-8')
    #sentencs_split_list=[['ab','a','c'],['a','b','c','c']]
    model = Word2Vec(sentencs_split_list, size=size, window=window, min_count=min_count, workers=workers)
    wv=model.wv
    #pathy=os.getcwd()+"\\wordvectors.txt"
    #wv.save(pathy)  
    #wv=KeyedVectors.load(pathy,mmap='r')
    del model
    
    index2word=wv.index2word
    k=1
    for wdx in index2word:
        ms=wv.most_similar(wdx,topn=topn)
        if k==1:
            most_sim=npy.concatenate((npy.array([wdx]*len(ms)).reshape(-1,1), npy.array(ms)),axis=1) 
            k=0
        else:
            tem=npy.concatenate((npy.array([wdx]*len(ms)).reshape(-1,1), npy.array(ms)),axis=1)
            most_sim=npy.concatenate((most_sim,tem),axis=0)
    rst=pds.DataFrame(most_sim,columns=['target_word','sim_word','w2v_similarity'])
    rst['w2v_similarity']=rst['w2v_similarity'].astype('float64')
    return rst.loc[rst.w2v_similarity>=sim_min,:].reset_index(drop=True)


def jaro(s1,s2,mode='winkler',h=3,p=0.1):
    '''
    s1,s2 : the text-string that will be calculate distince, length(s)>0
    mode :  ['popular' , 'improve' , 'winkler' , 'improve_winkler']
            popular  : use the most popular function
            improve[1,2] : use 't' in another way to make the function more reasonable 
            winkler : jaro-winkler  ,an improvement of jaro popular method [default]
            improve[1,2]_winkler : both improve and winkler
    h : int, if mode is winkler,active h ,which means head-str number less then h
    p : float, if mode is winkler,active p , must control h*p<=1.0 ,otherwise will raise error
        if None,set p=0.5/h 
    '''
    # 计算 距离小于param1的有效匹配到的字符MatchValue
    ls1=len(s1);ls2=len(s2)
    param1=max(max(ls1,ls2)/2.0-1.0,1.0) # 有效匹配的条件 ：不超过最大容忍距离,注意，最小也为1
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
    confx=SparkConf().setAppName('2_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")
    
    # 3 计算W2V相似度，获得最相似的topn个结果
    doc_table=params['shared']['doc_cut_word_table']
    df1=sqlContext.sql('select * from {0}.{1}'.format(database_name,doc_table))
    sts=df1.select('inputwds').toPandas()['inputwds'].values.tolist()
    ## 3.1 word2vec 余弦相似度,保留topn个
    sizex=params['shared']['sizex']
    windowx=params['shared']['windowx']
    topn=params['word2vec']['topn']
    sim_min=params['word2vec']['sim_min']
    word_sim=word2vec_most_sim(sentencs_split_list=sts,size=sizex, window=windowx, min_count=5, workers=4,topn=topn,sim_min=sim_min)  # 半小时
    ## 3.2 jaro improve3 文本相似度
    modex=params['jaro']['modex']
    word_sim['jaro_sim']=word_sim.apply(lambda x :jaro(s1=x[0],s2=x[1],mode=modex,h=3,p=0.1),axis=1)
    
    sim_df1=sqlContext.createDataFrame(word_sim)
    #sqlContext.sql('drop table if exists {0}.w2v_sim_on_product_name'.format(database_name)).collect()
    sim_df1.write.saveAsTable('{0}.w2v_sim_on_product_name'.format(database_name),mode='overwrite')
    del word_sim,sts
    gc.collect()
    
'''
word_sim.loc[word_sim.target_word==u'亮白',:]
word_sim.loc[word_sim['similarity']>=0.70,:].sort_values('similarity',ascending=False).head(100)
df1.filter(F.col('text_note').rlike('绞股蓝')).show()

tfidf_sim.loc[tfidf_sim.base_term==u'亮白',]
'''




















