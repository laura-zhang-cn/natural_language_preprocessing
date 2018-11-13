# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:21:04 2018

@author: zhangyaxu


商品描述/标题等数据源  -> 品类上的 合并
训练词的tfidf向量并计算离散向量相似度，返回topn个，并存储到hive中
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

import gc
import pandas as pds
import numpy as npy

from pyspark.sql import functions  as F
from pyspark.sql import Window

from gensim.models import TfidfModel
from gensim.corpora import Dictionary

from corpus_richness_alg import  term_doc_tf_r

from params import  params
database_name=params['shared']['database_name']

def term_tfidf_weight(model,corpus,na_val=0.0,keep_type='appeared',out_type='split',term_keep_index=None,norm_type=None):
    '''
    model is a TfidfModel
    corpus is a dictionary.doc2bow corpus which like [[(term_id,tf),..],..]  [docs=>(terms=>tf)]
    out_type : [split,aggr] return a datframe with vector as columns that split by doc_id or  aggregate in one columns 
    norm_type :[None,'maxscale']
    term_keep_index: list of term-id  which should keep ,if None,keep all
    return_val:['tfidf' ,'tf'], default 'tfidf'
    '''
    idf=pds.DataFrame(list(sorted(model.idfs.items())),columns=['term_id','idf'])
    def per_doc_wd_tfidf_weight(x,keep_type='appeared'):
        '''
        x is a document's words-corpus which is list type like [(term_id,tf),...]
        idf is term inverse-document-freq which is dataframe type like [(term_id,idf)]
        keep_type: ['appeared','all'] . appeared :让文档仅保留出现在文档中的词,all:保留所有词,即使没在文档中出现,对应的值是0.0
        return [(term_id,tf),..] 
        '''
        tf_r=npy.array(list(dict(x).values()))*1.0/sum(dict(x).values()) #  
        tf=pds.DataFrame(list(zip(dict(x).keys(),tf_r.tolist())),columns=['term_id','tf'])
        if keep_type=='all':
            how_type='left'
        else:
            how_type='inner'
        tf_idf=pds.merge(idf,tf,on='term_id',how=how_type).fillna(0.0)
        tf_idf['tfidf']=tf_idf['tf']*tf_idf['idf'] # 包含大于1的值
        tf_idf['tfidf']=tf_idf['tfidf']/tf_idf['tfidf'].max() #  去中心化 ,必须的，不然值都非常小
        #print(tf_idf.dtypes)
        return list(zip(tf_idf.term_id.values.tolist(),tf_idf.tfidf.values.tolist(),tf_idf.tf.values.tolist()))
    # 文章下的词tfidf值
    tfidf=list(map(lambda x : per_doc_wd_tfidf_weight(x=x,keep_type=keep_type),corpus))  # 文档层面的词向量表达 [[(term_id,tfidf),..],..]  [docs=>(terms=>tfidf)]
    ls_tfidf=[]
    doc_id=0
    # 展开文章，获得文章=>词=>tfidf值的dataframe 
    for doc_terms in tfidf: 
        ls_tfidf.extend([(doc_id,)+term_tfidf for term_tfidf in doc_terms]) 
        doc_id=doc_id+1
    del tfidf
    df_tfidf=pds.DataFrame(ls_tfidf,columns=['doc_id','term_id','tfidf','tf'])
    term_tfidf=df_tfidf.pivot(index='term_id',columns='doc_id',values='tfidf').fillna(na_val)  # 词 => tfidf_vectory
    if norm_type=='maxscale':
        term_tfidf=term_tfidf.apply(lambda x : x/x.max(),axis=1)  # 归一化
    #del df_tfidf
    #gc.collect()
    if term_keep_index!=None:
        term_tfidf=term_tfidf.loc[term_keep_index,:]
    if out_type=='aggr':
        #tfidf_vec=term_tfidf.apply(lambda x : x.tolist(),axis=1,result_type=None)    #  仅 python3 # python2 不支持 result_type=None,默认是split，所以不可行
        tfidf_vec=pds.DataFrame(list(zip(term_tfidf.index.values.tolist(),term_tfidf.values.tolist())),columns=['term_id','tfidf_vec']).set_index('term_id') #  python2 & python3 都适用，也很快
        return tfidf_vec
    elif out_type=='split':
        return term_tfidf
    else:
        return None


def tfidf_vec_generate_and_storage(cat_sts,na_val=0.0,norm_type='maxscale'):
    '''
    生成tfidf vec 并存储
    提前去掉低频词，仅保留高频词，
    对 稀疏向量 ，仅存储 非稀疏值 及对应位置，存储空间
    '''
    ## 训练模型
    dct = Dictionary(cat_sts) 
    
    corpus = [dct.doc2bow(line) for line in cat_sts]
    model = TfidfModel(corpus)
    tk2id = dct.token2id
    
    # 去掉低频词，仅保留非低频词 term_id_keep 会去索引 tfidf-vec
    tf_all=[]
    for c in corpus:
        tf_all.extend(c)
    tf_all=pds.DataFrame(tf_all,columns=['term_id','tf']).groupby('term_id')['tf'].agg(['sum','count']).reset_index().rename(columns={'sum':'tf','count':'df'})
    term_id_keep=tf_all.loc[tf_all.tf>3,:].term_id.values.tolist()
    sqlContext.createDataFrame(tf_all).write.saveAsTable('{0}.term_id_tf_df'.format(database_name),mode='overwrite')
    ## transform: 词的tf-idf向量 ,需要获得词的向量表达,一般是 对想基于词对文档进行分类，或者计算词的相似度 ,比较快，3到5分钟
    #term_tfidf=term_tfidf_weight(model=model,corpus=corpus,na_val=-1.0,keep_type='appeared',out_type='split')  # 获得词的tfidf向量表达，pandas-dataframe方式传出，一行是一个词在各doc中的tfidf权重向量 
    tfidf_vec=term_tfidf_weight(model=model,corpus=corpus,na_val=na_val,keep_type='appeared',out_type='aggr',term_keep_index=term_id_keep,norm_type=norm_type)
    #tfidf_vec=tfidf_vec
    del model  ,dct
    gc.collect()
    print(1)
    def sparse_vec_disassemble(a,sparse_val=-1.0,return_part=1):
        '''
        获取疏向量 非稀疏部分的位置p，最终二者分别保留p位的值
        a=npy.array([-1, -1,  0.2, 0.7, -1,  -1])
        则保留 
        a_u=[0.2,0.7] 
        return 保留向量
        '''
        pos_a=npy.array(range(len(a)))[a!=sparse_val].tolist()
        a_u=a[pos_a]
        if return_part==1:
            return a_u.tolist()
        elif return_part==2:
            return pos_a
        else:
            return a_u.tolist(),pos_a
    
    tfidf_vec2=tfidf_vec.apply(lambda x : sparse_vec_disassemble(a=npy.array(x[0]),sparse_val=na_val,return_part=1),axis=1).reset_index()
    tfidf_vec2['pos']=tfidf_vec.apply(lambda x : sparse_vec_disassemble(a=npy.array(x[0]),sparse_val=na_val,return_part=2),axis=1).reset_index(drop=True)
    tfidf_vec2.columns=['term_id','tfidf_vec','tfidf_vec_pos']
    print(2)
    #
    tfidf_vec_df=sqlContext.createDataFrame(tfidf_vec2)  # dataframe ，约10w条记录 ，rdd时计算
    windowx=Window.orderBy('term_id')
    tfidf_vec_df=tfidf_vec_df.withColumn('rankx',F.row_number().over(windowx))
    #
    #sqlContext.sql('drop table if exists {0}.term_id_tfidf_vec_in_cat '.format(database_name))
    tfidf_vec_df.write.saveAsTable('{0}.term_id_tfidf_vec_in_cat'.format(database_name),mode='overwrite')
    gc.collect()
    
    tk_id0=pds.DataFrame(list(tk2id.items()),columns=['term_name','term_id']) # dataframe 方便操作join
    tk_id=tk_id0.loc[(tk_id0.term_name.str.len()>1)&(tk_id0.term_name.str.contains(u'[\u4e00-\u9fa5a-zA-Z]')==True),:]  # 剔除长度为1 和 仅数字或符号的 词
    sqlContext.createDataFrame(tk_id).write.saveAsTable('{0}.term_id_name'.format(database_name),mode='overwrite')
    return corpus

if __name__=='__main__':
    confx=SparkConf().setAppName('3_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")
    
    ## 4 tfidf 相似度
    topic_table=params['shared']['topic_cut_word_table']
    df3=sqlContext.sql('select inputwds from {0}.{1}'.format(database_name,topic_table))
    # 4.1 生成tfidf 向量，并存储
    na_val=params['tfidf']['na_val']
    norm_type=params['tfidf']['norm_type']
    cat_sts=df3.select('inputwds').toPandas()['inputwds'].values.tolist()  # 2G #cat_sts=[['ab','a','c'],['a','b','c','c']]
    corpus=tfidf_vec_generate_and_storage(cat_sts,na_val=na_val,norm_type=norm_type) # NOT RETURN VALUE,BUT STORAGE TO TABLE : tfidf_vec_df ,tk_id
    gc.collect()
    
    # 4.2 计算相似度 建议这里重新开一个 py文件，这样可以释放内存 ，见 tfidf_vec_sim.py文件

    # 4.2 term_id tf ration in each doc --> used in richness_coef
    term_doc_tf_r(corpus=corpus,sqlContext=sqlContext)