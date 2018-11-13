# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:00:29 2018

@author: zhangyaxu

对于冷门类下，语料丰富度不够，导致训练出的w2v失真，w2v_sim出现假相似的情况，我们尝试设计语料丰富度系数 来衡量词的w2v_sim的可信度

如何定义 语料丰富度，如何计算可信度：

"""
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

from pyspark.sql import functions  as F
from pyspark.sql import Row as sp_row
#from pyspark.sql import Window

from ast import literal_eval

#import pandas as pds
import numpy as npy

from params import  params
database_name=params['shared']['database_name']

def term_doc_tf_r(corpus,sqlContext):
    '''
    corpus : 语料，一篇文章一个语料库 形如 ：[ [(term_id,tf),(term_id,tf),..] , [...] ],基于gensim.corpora.Dictionary  生成，参考tfidf_vec_generate中的使用，
    为了 doc-id 顺序在此处 和 tfidf_vec中存储的一致性，这个方法在 tfidf_vec_generate中调用，直接使用同一个corpus ，这样 doc-id 就一致了
    '''
    def tf_r(x):
        tf_sum=sum(dict(x).values())*1.0
        tf_avg=npy.array(dict(x).values()).mean()/tf_sum
        tf_std=npy.std(npy.array(dict(x).values())/tf_sum)
        return ([(xx[0],xx[1]/tf_sum) for xx in x], tf_sum ,float(tf_avg) ,float(tf_std))  # 比使用zip(dict.keys,array()/sum )的矢量运算的方式慢，但是 转化为DataFrame不会报tuple不支持的错误，晓不得为什么zip形成的tuple会报错
    tf_r_in_doc=list(map(lambda x: tf_r(x),corpus))
    tf_r_in_doc20=[]
    tf_r_in_doc21=[]
    tf_r_in_doc22=[]
    tf_r_in_doc23=[]
    for  xx in tf_r_in_doc:
        tf_r_in_doc20.append(str(xx[0]))
        tf_r_in_doc21.append(xx[1])
        tf_r_in_doc22.append(xx[2])
        tf_r_in_doc23.append(xx[3])
    tf_r_in_doc3=list(zip(list(range(0,len(tf_r_in_doc))),tf_r_in_doc20,tf_r_in_doc21,tf_r_in_doc22,tf_r_in_doc23)) # key is doc-id ,value is corpus-tf-r  : [(doc_id,[(term_id,tf_r),(term_id,tf_r),..]) ,...}
    df=sqlContext.createDataFrame(tf_r_in_doc3,['doc_id','term_tf_r','doc_tf','term_tf_r_avg','term_tf_r_std'])
    sqlContext.sql('drop table if exists {0}.doc_term_tf_r'.format(database_name))
    df.write.saveAsTable('{0}.doc_term_tf_r'.format(database_name),mode='overwrite')
    return 'tf_r_in_doc over'

def data_etl(sqlContext,table_list,windowx=3):
    ## rel coef data
    #windowx=3
    avg_rel_thr=1.0/windowx**(windowx/2.0)
    rel_table=table_list[0]
    term_rel=sqlContext.sql('select * from {0} where avg_relevance>{1}'.format(rel_table,avg_rel_thr))
    term_rel_total=term_rel.filter(F.col('flag_word')==F.col('target_word')).selectExpr('flag_word','total_comment_num as total_doc_num')
    term_rel_co=term_rel.filter(F.col('flag_word')!=F.col('target_word')).selectExpr('flag_word','target_word','total_comment_num as co_doc_num ')
    # reduce join
    term_rel_co_ratio=term_rel_co.join(term_rel_total,'flag_word','inner').withColumn('co_doc_ratio',F.col('co_doc_num')*1.0/F.col('total_doc_num')) # 这个join可能导致数据倾斜,后面发现不是这里。
    #
    term_id_name_table=table_list[1]
    term_id_name=sqlContext.sql('select * from {0}'.format(term_id_name_table))
    term_rel_co_ratio=term_rel_co_ratio.join(term_id_name,term_rel_co_ratio.flag_word==term_id_name.term_name,'inner').drop('term_name').withColumnRenamed('term_id','flag_term_id')
    term_rel_co_ratio=term_rel_co_ratio.join(term_id_name,term_rel_co_ratio.target_word==term_id_name.term_name,'inner').drop('term_name').withColumnRenamed('term_id','target_term_id')
    ## tfidf-vec-doc
    tfidf_vec_table=table_list[2]
    tfidf_vec_doc=sqlContext.sql("select term_id,tfidf_vec_pos as appear_docs from {0}".format(tfidf_vec_table))
    #
    co_ratio_appear_docs=term_rel_co_ratio.join(tfidf_vec_doc,term_rel_co_ratio.flag_term_id==tfidf_vec_doc.term_id,'inner').drop('term_id').withColumnRenamed('appear_docs','flag_term_docs')
    co_ratio_appear_docs=co_ratio_appear_docs.join(tfidf_vec_doc,co_ratio_appear_docs.target_term_id==tfidf_vec_doc.term_id,'inner').drop('term_id').withColumnRenamed('appear_docs','target_term_docs')
    #['flag_word', 'target_word', 'co_doc_num', 'total_doc_num', 'co_doc_ratio', 'flag_term_id', 'target_term_id', 'flag_term_docs', 'target_term_docs']
    co_ratio_appear_docs=co_ratio_appear_docs.select('flag_term_id','target_term_id','co_doc_ratio','flag_term_docs','target_term_docs')
    ## doc term tf r
    doc_tf_table=table_list[3]
    doc_tf_r=sqlContext.sql('select * from {0}'.format(doc_tf_table))
    return co_ratio_appear_docs,doc_tf_r


def rel_term_doc_freq_coef(m_x,testx=False):
    '''
    :param m_x:  a Row-type / tuple-type value or  a list Of Row-type value
    :param testx: [False,True] means whether debug ,defalt False : not debug
    :return:  a tuple type result
    '''
    def a_row_cal(x,s_w=1.5):
        '''
        :param x: a base value : Row-type or tuple-type
        :return:  a float which means x[1] term's  freq_coef
        '''
        doc_tf_r=doc_tf_r_bd.value
        inter_docs=x[3] # inter_docs=list(set(x[3]).intersection(set(x[4]))) # 公共的doc,提前处理了这两个数组类型的值，避免数据的倾斜情况 （因为两个数组的大小和内部值都不是很均衡）
        # doc_tf_r为pandas dataframe类型，索引获取多个key值效率高
        term_tf_r=doc_tf_r.loc[inter_docs,'term_tf_r_dict'].values.tolist()
        term_tf_r_avg=doc_tf_r.loc[inter_docs,'term_tf_r_avg'].values
        doc_tf=doc_tf_r.loc[inter_docs,'doc_tf'].values
        term_tf_r_std=doc_tf_r.loc[inter_docs,'term_tf_r_std'].values
        if testx==False:
            # 关联的那个词在各公共doc的tf_r
            # 最终检查发现，literal_eval这一句的效率太低了，影响比未对常量进行broadcast还要大，所以在map调用之前先处理了
            # 提前转化好了dict(literal_eval(some_format_str))的结果 ，直接应用 tf_r
            tf_r_ls=list(map(lambda tf_r :tf_r.get(x[1]),term_tf_r))
        else:
            term_tf_r_avg=[0.1,0.3,0.05]
            doc_tf=[123,324,879]
            tf_r_ls=[0.2,0.34,0.02]
        Weithtx=npy.array(doc_tf)*1.0/sum(doc_tf)
        tf_r_avg=npy.sum(Weithtx*(npy.array(term_tf_r_avg)+s_w*npy.array(term_tf_r_std))) # 加权总离散平均值
        tf_r=npy.sum(Weithtx* npy.array(tf_r_ls)) # 加权平均值
        freq_coef= float(tf_r/tf_r_avg)   # 与离散值比较
        return freq_coef
    if type(m_x)==type(sp_row()) or type(m_x)==type(tuple()):
        # map ,get  'Row' type
        rst=(m_x[0],m_x[1],m_x[2],a_row_cal(x=m_x))
    else:
        #mapPartitions ,get 'itertools.chain'  type
        rst=[]
        for y in m_x:
            rst.append((y[0],y[1],y[2],a_row_cal(x=y)))
    return rst


if __name__=='__main__':
    confx=SparkConf().setAppName('3_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")

    # 1 data read and etl
    ## 1.1 base read and join
    windowx=params['shared']['windowx'] # 与w2v保持一致
    sizex=params['shared']['sizex'] # 与w2v保持一致
    table_list=[
        '{0}.effect_words_relevance_product_name'.format(database_name),
        '{0}.term_id_name'.format(database_name),
        '{0}.term_id_tfidf_vec_in_cat '.format(database_name),
        '{0}.doc_term_tf_r'.format(database_name)
        ]

    co_ratio_appear_docs, doc_tf_r = data_etl(sqlContext,table_list=table_list,windowx=windowx)
    ## 1.2 more etl
    co_ratio_appear_docs2=co_ratio_appear_docs.rdd.map(lambda x:x[0:3]+(list(set(x[3]).intersection(set(x[4]))),)).filter(lambda x: len(x[3])>0) # 提前处理， 需要计算交集的两个数组，避免数据的倾斜（不提前处理 是存在倾斜的）
    doc_tf_r_pd=doc_tf_r.toPandas().set_index('doc_id')  # pandas dataframe
    doc_tf_r_pd['term_tf_r_dict']=doc_tf_r_pd.term_tf_r.apply(lambda tf_r :dict(literal_eval(tf_r))) # 提前处理 str(list(tuple))-type into dict-type ，不在map中处理
    del doc_tf_r_pd['term_tf_r']
    doc_tf_r_bd=sc.broadcast(doc_tf_r_pd) # 必须broadcast ，一个50M左右大小的常量 传递过程的通信时间太长了，每个task都要等待很久，造成整体效率极低，另外 broadcast的传递方式 不要以常参的方式传入函数，而是直接调用即可。

    # 2  AB组合的两两计算 获得每个关联词B的频繁度
    co_ratio_freq_coef=co_ratio_appear_docs2.map(lambda x : rel_term_doc_freq_coef(x,testx=False))
    co_ratio_freq_coef_df=sqlContext.createDataFrame(co_ratio_freq_coef,['flag_term_id','target_term_id','co_doc_ratio','target_term_freq_coef'])
    rela_term_table='{0}.rela_term_freq_coef'.format(database_name)
    co_ratio_freq_coef_df.write.saveAsTable(rela_term_table,mode='overwrite')

    # 3  获得 词A的上下文语料的总单调性，即 在A指定的上下文窗口window内 ，包含的高频繁度的关联词的比例p 越高 和 关联词的频繁度f 越高，则词A的上下文语料的单调性越高==上下文的语料丰富度越低
    rela_term_table='{0}.rela_term_freq_coef'.format(database_name)
    co_ratio_freq_coef_df=sqlContext.sql('select * from {0}'.format(rela_term_table))
    p='co_doc_ratio'
    f='target_term_freq_coef'
    context_num=params['shared']['sizex'] # 认为 上下文的合理词数量
    term_richness_coef=co_ratio_freq_coef_df.groupBy('flag_term_id').agg((F.sum(F.when(F.col(f)>=1,F.col(p)).otherwise(0.0))/F.sum(F.col(p))).alias('p_coef'),
                                                                         F.log2(F.sum(F.when(F.col(f) >= 1, F.col(p)).otherwise(0.0)*F.when(F.col(f) >= 1, F.col(f)).otherwise(0.0))+2.0).alias('pf_weight'),
                                                                         (F.avg(F.col(p))*1.0/0.1).alias('p_over_thr'),
                                                                         (context_num*1.0/F.count(F.col(p))).alias('context_num_coef')
                                                                         ).withColumn('unrichness_coef',F.col('p_coef')*F.col('pf_weight'))
    sqlContext.sql('drop table if exists {0}.prod_name_word_term_unrichness_coef'.format(database_name))
    term_richness_coef.write.saveAsTable('{0}.prod_name_word_term_unrichness_coef'.format(database_name),mode='overwrite')







