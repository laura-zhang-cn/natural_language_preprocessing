# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:58:02 2018

@author: zhangyaxu
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

from pyspark.sql import functions  as F

#import os
import pandas as pds
import numpy as npy

from params import  params
database_name=params['shared']['database_name']

def basic_feature_get(x):
    """
    :param x:  a SPARK-SQL - Row type
    :return:  tuple ,  new features that used in algorithm
    """
    tf_max=max(x.tgt_tf,x.sim_tf)
    tf_min=min(x.tgt_tf,x.sim_tf)
    df_max=max(x.tgt_df,x.sim_df)
    if x.flag_flag != None:
        flag_cond = len(set(x.flag_flag).intersection(set(x.target_flag))) * 1.0 / max(len(x.flag_flag),len(x.target_flag))  # flag 相似度
        co_occurence_num = x.final_relevance_coef * 1.0 / x.avg_relevance ** 2  # 共现doc数
        co_occurrence_cond = co_occurence_num / (min(x.tgt_tf, x.sim_tf) / 5.0)  # 共现度系数
    else:
        flag_cond = 0.0
        co_occurence_num = 0.0
        co_occurrence_cond = 0.0
    p_coef_max=max(x.p_coef1,x.p_coef2)
    pf_weight_max=max(x.pf_weight1,x.pf_weight2)
    p_over_thr1_max=max(x.p_over_thr1,x.p_over_thr2)
    context_num_coef_max=max(x.context_num_coef1,x.context_num_coef2)
    unrichness_coef_max = max(x.unrichness_coef1, x.unrichness_coef2)
    p_coef_min=min(x.p_coef1,x.p_coef2)
    pf_weight_min=min(x.pf_weight1,x.pf_weight2)
    p_over_thr1_min=min(x.p_over_thr1,x.p_over_thr2)
    context_num_coef_min=min(x.context_num_coef1,x.context_num_coef2)
    unrichness_coef_min = min(x.unrichness_coef1, x.unrichness_coef2)
    rst=(
        x.target_word,
        x.sim_word,
        x.w2v_similarity,
        x.jaro_sim,
        x.tfidf_similarity,
        tf_max,
        tf_min,
        df_max,
        x.tfidf_plus_w2v,
        x.tfidf_plus_jaro,
        flag_cond,
        x.avg_relevance if x.avg_relevance>0 else 0.0,
        co_occurence_num,
        co_occurrence_cond,
        p_coef_max,
        pf_weight_max,
        p_over_thr1_max,
        context_num_coef_max,
        unrichness_coef_max,
        p_coef_min,
        pf_weight_min,
        p_over_thr1_min,
        context_num_coef_min,
        unrichness_coef_min
    )
    return rst

if __name__=='__main__':
    confx=SparkConf().setAppName('6_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")

    # 1 各方数据读取 和join  .  w2v tfidf jaro relevance tf unrichness
    w2v_sim=sqlContext.sql('select * from {0}.w2v_sim_on_product_name'.format(database_name)) # 出现过的词都会有一个w2v向量
    tfidf_all_sim=sqlContext.sql('select * from {0}.tfidf_sim_all_on_product_name'.format(database_name)) # 出现过的词 都会有一个 tfidf向量
    #tfidf_sim=sqlContext.sql('select * from {0}.tfidf_sim_on_product_name'.format(database_name))
    sqlx1='''
    SELECT 
    tb2.term_name,tb1.* 
    FROM {0}.term_id_tf_df tb1 
    JOIN {0}.term_id_name tb2 
    ON tb1.term_id=tb2.term_id 
    '''.format(database_name)
    wd_tf_total=sqlContext.sql(sqlx1) # 出现过的词 都有词频 和文档数

    relevance=sqlContext.sql('select flag_word as target_word,target_word as sim_word,flag_flag,target_flag,avg_relevance,final_relevance_coef from {0}.effect_words_relevance_product_name'.format(database_name)) # 并非同义词都能 会同时出现在一个描述中过，所以 这个数据 应用时请被left join

    sqlx2='''
    SELECT 
    tb2.term_name ,
    tb1.p_coef,
    tb1.pf_weight,
    tb1.p_over_thr,
    tb1.context_num_coef,
    tb1.unrichness_coef
    from {0}.prod_name_word_term_unrichness_coef tb1
    JOIN {0}.term_id_name tb2 ON tb1.flag_term_id=tb2.term_id
    '''.format(database_name)
    unrichness=sqlContext.sql(sqlx2)

    # 2 基础筛选 和 融合，初步收缩数据规模。
    w2v_thr = 0.7
    tfidf_thr1 = 0.55

    sim_join = w2v_sim.filter(F.col('w2v_similarity') >= w2v_thr).join(tfidf_all_sim.filter(F.col('tfidf_similarity') >= tfidf_thr1), ['target_word', 'sim_word'], 'inner')
    sim_join = sim_join.join(wd_tf_total.filter(F.col('tf') > 10).select('term_name', 'tf', 'df'),sim_join.target_word == wd_tf_total.term_name, 'inner').drop('term_name').withColumnRenamed('tf', 'tgt_tf').withColumnRenamed('df', 'tgt_df')
    sim_join = sim_join.join(wd_tf_total.filter(F.col('tf') > 10).select('term_name', 'tf', 'df'),sim_join.sim_word == wd_tf_total.term_name, 'inner').drop('term_name').withColumnRenamed('tf', 'sim_tf').withColumnRenamed('df', 'sim_df')

    sim_join = sim_join.withColumn('tfidf_plus_w2v', F.col('w2v_similarity') + F.col('tfidf_similarity')).withColumn('tfidf_plus_jaro', F.col('jaro_sim') + F.col('tfidf_similarity'))

    sim_join = sim_join.join(relevance, ['target_word', 'sim_word'], 'left')
    #['term_name', 'p_coef', 'pf_weight', 'p_over_thr', 'context_num_coef', 'unrichness_coef']
    unrichness1=unrichness.withColumnRenamed('p_coef','p_coef1').withColumnRenamed('pf_weight','pf_weight1').withColumnRenamed('p_over_thr','p_over_thr1').withColumnRenamed('context_num_coef','context_num_coef1').withColumnRenamed('unrichness_coef','unrichness_coef1')
    unrichness2=unrichness.withColumnRenamed('p_coef','p_coef2').withColumnRenamed('pf_weight','pf_weight2').withColumnRenamed('p_over_thr','p_over_thr2').withColumnRenamed('context_num_coef','context_num_coef2').withColumnRenamed('unrichness_coef','unrichness_coef2')
    sim_join=sim_join.join(unrichness1,sim_join.target_word==unrichness1.term_name,'inner').drop('term_name')
    sim_join=sim_join.join(unrichness2,sim_join.sim_word==unrichness2.term_name,'inner').drop('term_name')

    mix_model_data_rdd=sim_join.rdd.map(lambda x :basic_feature_get(x=x))  # 这样初步筛选后，可降低样本总量

    columns_new=[
        'target_word',
        'sim_word',
        'w2v_similarity',
        'jaro_sim',
        'tfidf_similarity',
        'tf_max',
        'tf_min',
        'df_max',
        'tfidf_plus_w2v',
        'tfidf_plus_jaro',
        'flag_cond',
        'avg_relevance',
        'co_occurence_num',
        'co_occurrence_cond',
        'p_coef_max',
        'pf_weight_max',
        'p_over_thr1_max',
        'context_num_coef_max',
        'unrichness_coef_max',
        'p_coef_min',
        'pf_weight_min',
        'p_over_thr1_min',
        'context_num_coef_min',
        'unrichness_coef_min'
        ]

    mix_model_data_df=sqlContext.createDataFrame(mix_model_data_rdd,columns_new)
    mix_model_data_df.write.saveAsTable('{0}.synonyms_mix_model_data'.format(database_name),mode='overwrite') # 将整合的子模型结果作为feature， 存到数据库



