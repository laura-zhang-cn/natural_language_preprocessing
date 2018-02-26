import jieba
from  jieba.posseg import cut
import pandas as pds
import re
import numpy as npy
import math
import sys 

'''
文本相关算法 一直苦于数量巨大 导致计算慢，
所以，这里与spark集群结合，使用rdd加快计算效率
无spark集群的需要改rdd那部分
'''
from pyspark.sql import Row as sp_row
from pyspark.storagelevel import StorageLevel
from  pyspark import SparkConf
from  pyspark import SparkContext
from  pyspark.sql import  HiveContext

from pyspark.sql import functions as func
from pyspark.sql import Window




def words_relevance(words_specific_df,sqlContext):
    '''
    words_specific_df : 需要计算距离的目标词 , pandas dataframe ,来自 wd_emo 取了其中两列 ['word' ,'emo_promote_coef']
    '''
    # 1 extract word column 
    target_word_df=words_specific_df[['word']].copy()
    # 2 get word-comments_ cut result ,这里是从数据库拿出之前分完词的结果
    comments_words_new=sqlContext.sql("select comment_id,comments_words_new from recommend.comments_cut_effect_words_result_new ")
    # 3 calculate relevance in each comment
    def words_relevance_one_comment(comment_split_words,target_word_df=target_word_df):
        '''
        计算单个评价句子中 强促进目标词和有效词性的词的上下文关联度
        单个评价 示例 comment_split_words=[{'word':'物流','flag':'n'},{'word':'不满意','flag':'dv'},{'word':'物流','flag':'n'},{'word':'太慢','flag':'da'},{'word':'拿着','flag':'vu'},{'word':'质量','flag':'n'}]
        return relevance_in_each_comment  每个句子中 有效flag词和目标target词的关联度
        '''
        if type(comment_split_words)==type(sp_row()):
            idx=comment_split_words['comment_id']
            comment_split_words=comment_split_words['comments_words_new']
        effect_flag_weight=pds.DataFrame([('n',1),('ns',1),('nt',1),('nz',1),('nr',0.5),
                                          ('nv',0.67),('vn',0.67),('an',0.67) 
                                          ],columns=['flag','weight'])
        if len(comment_split_words)>0 :
            comment_words_df=pds.DataFrame(comment_split_words)
            comment_words_df=comment_words_df.loc[(comment_words_df.word.isin(target_word_df.word))|(comment_words_df.flag.isin(effect_flag_weight.flag)),:].reset_index(drop=True).reset_index().rename(columns={'index':'position'})
            flag_words=pds.merge(comment_words_df,effect_flag_weight,on='flag',how='inner',sort=False).rename(columns={'position':'flag_pos','flag':'flag_flag','word':'flag_word'})
            target_words=pds.merge(comment_words_df,target_word_df,on='word',how='inner',sort=False).rename(columns={'position':'target_pos','flag':'target_flag','word':'target_word'})
            if flag_words.shape[0]>0 and target_words.shape[0]>0:
                #目标词和有效词的df 求笛卡尔积
                flag_words['dikar_id']=0
                target_words['dikar_id']=0
                merge_df=pds.merge(flag_words,target_words,on='dikar_id')
                # tips ：若词与自己进行计算 ,先加一个值0.9，避免求出 inf 
                merge_df.loc[merge_df.flag_pos==merge_df.target_pos,'target_pos']=merge_df.loc[merge_df.flag_pos==merge_df.target_pos,'target_pos']+0.9 
                merge_df['each_relevance']=merge_df.weight/(npy.abs(merge_df.flag_pos-merge_df.target_pos))
                # 一个句子中，目标词和有效词只能有一个 关联度 ，当存在重复多个时，取单个句子内的最大关联。
                merge_df_group=merge_df.groupby(['flag_word','target_word'])['each_relevance'].max().reset_index()
                merge_df_group['comment_id']=idx
                relevance_in_each_comment=merge_df_group.values.tolist()
            else:
                relevance_in_each_comment=[]
        else:
            relevance_in_each_comment=[]
        return relevance_in_each_comment
    relevace_in_each_comment=comments_words_new.rdd.flatMap(words_relevance_one_comment) #
    relevace_in_each_comment_df=sqlContext.createDataFrame(relevace_in_each_comment,['flag_word','target_word','each_relevance','comment_id'])
    relevace_in_each_comment_df.persist(StorageLevel(True,True,False,False,1))
    relevace_in_each_comment_df.write.saveAsTable('recommend.effect_flag_words_relevance_in_each_comment',mode='overwrite')
    # 4 calculate summary_relevance  
    effect_words_relevance_tem=relevace_in_each_comment_df.groupBy(['flag_word','target_word']).agg(func.sum('each_relevance'),func.count('each_relevance'))
    effect_words_relevance_tem=effect_words_relevance_tem.withColumnRenamed('sum(each_relevance)','sum_relevance').withColumnRenamed('count(each_relevance)','total_comment_num')
    effect_words_relevance=effect_words_relevance_tem.withColumn('final_relevance_coef',func.pow(effect_words_relevance_tem.sum_relevance,2)/effect_words_relevance_tem.total_comment_num)
    # 4.2 rank in partition 
    windw=Window.partitionBy('flag_word').orderBy(func.desc('final_relevance_coef'))
    effect_words_relevance=effect_words_relevance.select('*',func.rank().over(windw).alias('rank_in_flag_word'))
    # 4.3 append target word emo_promote_coef
    words_specific_df=sqlContext.createDataFrame(words_specific_df)
    effect_words_relevance=effect_words_relevance.join(words_specific_df,effect_words_relevance.target_word==words_specific_df.word).select('flag_word','target_word','sum_relevance','total_comment_num','final_relevance_coef','rank_in_flag_word','emo_promote_coef')
    effect_words_relevance.persist(StorageLevel(True,True,False,False,1))
    no_out=sqlContext.sql('drop table recommend.effect_flag_words_relevance')
    effect_words_relevance.write.saveAsTable('recommend.effect_flag_words_relevance',mode='overwrite')
    return effect_words_relevance
    
    
    
if __name__=='__main__':
    #初始化
    confx=SparkConf().setAppName('implement  words relevance based on context ')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")
    
    words_specific_df=sqlContext.sql('select word ,emo_promote_coef from recommend.word_emo').toPandas() # convert to pandas dataframe
    # effect_words_relevance : a spark sql dataframe
    effect_words_relevance=words_relevance(words_specific_df,sqlContext)
