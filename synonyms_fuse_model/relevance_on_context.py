# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:10:26 2018

@author: zhangyaxu
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

import pandas as pds
import numpy as npy
#import math
#import sys 
#import re
#from operator import add
#from collections import Counter

from pyspark.sql import functions  as F
from pyspark.sql import Row as sp_row
from pyspark.sql import Window
#from pyspark.storagelevel import StorageLevel

from params import params

database_name=params['shared']['database_name']

def words_relevance(words_specific_df,sqlContext,high_type='all'):
    '''
    words_specific_df : 需要计算距离的目标词 , pandas dataframe  
    '''
    if high_type=='all':
        words_specific_df=None
    coef_col_name=''
    storage_table1='{0}.effect_words_relevance_product_name'.format(database_name)
    storage_table2='{0}.effect_words_relevance_in_each_product_name'.format(database_name)
    # 1 extract word column 
    if words_specific_df!=None:
        target_word_df=words_specific_df[['word']].copy()
    else:
        target_word_df=None
    # 2 get word-comments_ cut result
    idx_name='product_id'
    doc_table=params['shared']['doc_cut_word_table']
    comments_words_new=sqlContext.sql("select product_id as {0},cut_word_flag as comments_words_new from {1}.{2}".format(idx_name,database_name,doc_table))
    # 3 calculate relevance in each comment
    def words_relevance_one_comment(comment_split_words,target_word_df=target_word_df):
        '''
        计算单个评价句子中 目标词和有效词性的词的上下文关联度
        目标词 target_word_df 不传入时 ， 则计算句子中每个词之间的上下文关联度（当前默认）
        单个评价 示例 comment_split_words=[{'word':'不用','flag':'v'},{'word':'担心','flag':'v'},{'word':'简约','flag':'a'},{'word':'时尚','flag':'an'},{'word':'拿着','flag':'vu'},{'word':'质量','flag':'n'}]
        return relevance_in_each_comment  每个句子中  词的关联度
        '''
        if type(comment_split_words)==type(sp_row()):
            idx=comment_split_words[idx_name]
            comment_split_words=comment_split_words['comments_words_new'] 
        effect_flag_weight=pds.DataFrame([('n',1),('ns',1),('nt',1),('nr',0.5),('nz',0.67), 
                                          ('nv',0.67),('vn',0.67),('an',0.67) 
                                          ],columns=['flag','weight']) 
        if len(comment_split_words)>0 : 
            comment_words_df=pds.DataFrame(comment_split_words) 
            
            if target_word_df!=None:
                comment_words_df=comment_words_df.loc[(comment_words_df.word.isin(target_word_df.word))|(comment_words_df.flag.isin(effect_flag_weight.flag)),:].reset_index().rename(columns={'index':'position'})
                flag_words=pds.merge(comment_words_df,effect_flag_weight,on='flag',how='inner',sort=False).rename(columns={'position':'flag_pos','flag':'flag_flag','word':'flag_word'})
                target_words=pds.merge(comment_words_df,target_word_df,on='word',how='inner',sort=False).rename(columns={'position':'target_pos','flag':'target_flag','word':'target_word'})
            else:
                comment_words_df=comment_words_df.reset_index().rename(columns={'index':'position'})
                flag_words=comment_words_df.copy();flag_words['weight']=1.0;flag_words.rename(columns={'position':'flag_pos','flag':'flag_flag','word':'flag_word'},inplace=True)
                target_words=comment_words_df.copy();target_words.rename(columns={'position':'target_pos','flag':'target_flag','word':'target_word'},inplace=True)
            if flag_words.shape[0]>0 and target_words.shape[0]>0:
                #目标词和有效词的df 求笛卡尔积
                flag_words['dikar_id']=0
                target_words['dikar_id']=0
                merge_df=pds.merge(flag_words,target_words,on='dikar_id')
                # tips ：若词与自己进行计算 ,先加一个值0.9，避免求出 inf 
                merge_df.loc[merge_df.flag_pos==merge_df.target_pos,'target_pos']=merge_df.loc[merge_df.flag_pos==merge_df.target_pos,'target_pos']+0.9 
                merge_df['pos_diff']=npy.abs(merge_df.flag_pos-merge_df.target_pos)
                merge_df['each_relevance']=merge_df.weight/(npy.power(merge_df.pos_diff,merge_df.pos_diff/2.0)) # 使上下文距离的 衰减速度 随距离增大而增大
                # 一个句子中，目标词和有效词只能有一个 关联度 ，当存在重复多个时，取单个句子内的最大关联。
                merge_df_group=merge_df.groupby(['flag_word','target_word'])['each_relevance'].max().reset_index()
                merge_df=merge_df.drop_duplicates(subset=['flag_word','target_word'])[['flag_word','target_word','flag_flag','target_flag']]
                merge_df_group=pds.merge(merge_df_group.copy(),merge_df,on=['flag_word','target_word'])
                merge_df_group[idx_name]=idx
                relevance_in_each_comment=merge_df_group.values.tolist()
            else:
                relevance_in_each_comment=[]
        else:
            relevance_in_each_comment=[]
        return relevance_in_each_comment
    relevance_in_each_comment=comments_words_new.rdd.flatMap(words_relevance_one_comment) # 
    relevance_in_each_comment_df=sqlContext.createDataFrame(relevance_in_each_comment,['flag_word','target_word','each_relevance','flag_flag','target_flag',idx_name])
    #relevance_in_each_comment_df.persist(StorageLevel(True,True,False,False,1))
    #no_out=sqlContext.sql('drop table if exists {0}'.format(storage_table2)).collect()
    relevance_in_each_comment_df.write.saveAsTable('{0}'.format(storage_table2),mode='overwrite') # 80 executor 很快。。
    relevance_in_each_comment_df=sqlContext.sql('select * from {0}'.format(storage_table2))
    # 4 calculate summary_relevance  
    effect_words_relevance_tem=relevance_in_each_comment_df.groupBy(['flag_word','target_word']).agg(F.sum('each_relevance'),F.count('each_relevance'),F.avg('each_relevance'))
    effect_words_relevance_tem=effect_words_relevance_tem.withColumnRenamed('sum(each_relevance)','sum_relevance').withColumnRenamed('count(each_relevance)','total_comment_num').withColumnRenamed('avg(each_relevance)','avg_relevance')
    effect_words_relevance=effect_words_relevance_tem.withColumn('final_relevance_coef',F.pow(effect_words_relevance_tem.sum_relevance,2)/effect_words_relevance_tem.total_comment_num)
    # 4.2 rank in partition 
    windw=Window.partitionBy('flag_word').orderBy(F.desc('final_relevance_coef'))
    effect_words_relevance=effect_words_relevance.select('*',F.rank().over(windw).alias('rank_in_flag_word'))
    # 4.3 append word_flag
    words_flag_tem=relevance_in_each_comment_df.drop_duplicates(['flag_word','target_word']).select('flag_word','target_word','flag_flag','target_flag')
    effect_words_relevance=effect_words_relevance.join(words_flag_tem,['flag_word','target_word'])
    # 4.4 append target word emo_promote_coef
    if high_type!='all':
        words_specific_df=sqlContext.createDataFrame(words_specific_df)
        effect_words_relevance=effect_words_relevance.join(words_specific_df,effect_words_relevance.target_word==words_specific_df.word).select('flag_word','target_word','flag_flag','target_flag','sum_relevance','total_comment_num','avg_relevance','final_relevance_coef','rank_in_flag_word',coef_col_name)
    else:
        effect_words_relevance=effect_words_relevance.select('flag_word','target_word','flag_flag','target_flag','sum_relevance','total_comment_num','avg_relevance','final_relevance_coef','rank_in_flag_word')
    #effect_words_relevance.persist(StorageLevel(True,True,False,False,1))
    #no_out=sqlContext.sql('drop table if exists {0}'.format(storage_table1)).collect()
    effect_words_relevance.write.saveAsTable('{0}'.format(storage_table1),mode='overwrite')
    return 'effect_words_relevance run over'


if __name__=='__main__':
    confx=SparkConf().setAppName('5_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")
    
    words_relevance(words_specific_df=None,sqlContext=sqlContext,high_type='all')



















