# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:25:33 2018

@author: zhangyaxu

商品描述/标题等数据源  ,进行切词，并整合成 后续可使用的格式 存储到hive中
"""


from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

from  jieba.posseg import cut
import jieba
#import os

import math
import sys 
import re

from pyspark.sql import functions  as F
from pyspark.sql import Row as sp_row

from params import params
database_name=params['shared']['database_name']

def data_collect(sqlContext):
    sqlx='''
    SELECT 
    product_id,
    concat_ws(',',
        (CASE WHEN description_properties='NULL' or description_properties='' THEN NULL ELSE description_properties END),
        (CASE WHEN sell_note='NULL' or sell_note=''  THEN NULL ELSE sell_note END),  
        (CASE WHEN product_note='NULL' or product_note='' THEN NULL ELSE product_note END), 
        (CASE WHEN product_short_name='NULL' or product_short_name='' THEN NULL ELSE product_short_name END), 
        (CASE WHEN product_long_name='NULL' or product_long_name='' THEN NULL ELSE product_long_name END),
        (CASE WHEN product_medium_name='NULL' or product_medium_name='' THEN NULL ELSE product_medium_name END)
        ) as  text_note
    from mysql.jumei_product_detail 
    WHERE
    product_id is NOT NULL 
        AND length(concat_ws('',
        (CASE WHEN description_properties='NULL' or description_properties='' THEN NULL ELSE description_properties END),
        (CASE WHEN sell_note='NULL' or sell_note=''  THEN NULL ELSE sell_note END),
        (CASE WHEN product_note='NULL' or product_note='' THEN NULL ELSE product_note END),
        (CASE WHEN product_short_name='NULL' or product_short_name='' THEN NULL ELSE product_short_name END),
        (CASE WHEN product_long_name='NULL' or product_long_name='' THEN NULL ELSE product_long_name END),
        (CASE WHEN product_medium_name='NULL' or product_medium_name='' THEN NULL ELSE product_medium_name END)
        ))>=4
    '''
    df0=sqlContext.sql(sqlx)
    
    # 仅 保留 近一年有售出 且
    #sqly='''
    #SELECT distinct product_id as product_id FROM bi_datawarehouse.int_all_orders 
    #WHERE data_date>date_sub(current_date,720) 
    #AND product_id>0 
    #'''
    #df_sales_prod=sqlContext.sql(sqly)
    #df0=df0.join(df_sales_prod,on='product_id',how='inner')
    
    # 保留 60天内曾在售的商品
    sqly1='''SELECT distinct product_id as pd1 from mysql.jumei_mall WHERE status in(1) '''
    sqly2='''SELECT distinct product_id as pd2 from mysql.jumei_mall_products WHERE show_status in(1,6) '''
    sqly3='''SELECT distinct product_id as pd3 from mysql.tuanmei_deals WHERE status in(0,1) and end_time>=unix_timestamp()-5184000 '''
    
    y1=sqlContext.sql(sqly1)
    y2=sqlContext.sql(sqly2)
    y3=sqlContext.sql(sqly3)
    
    df0=df0.join(y1,df0.product_id==y1.pd1,how='left'
                 ).join(y2,df0.product_id==y2.pd2,how='left'
                 ).join(y3,df0.product_id==y3.pd3,how='left')
    df0=df0.filter('pd1 is not null or pd2 is not null or pd3 is not null').drop('pd1').drop('pd2').drop('pd3')
    pcat=sqlContext.sql('SELECT product_id,category_v3_4 as cat_id from mysql.tuanmei_products')
    df0=df0.join(pcat,'product_id','inner')
    return df0


def concat_prop_words(cut_iter,text,text_id):
    #1 由于python2.7 的默认encoding是 ascii , 与unicode存在兼容问题，遂需要设置一下。 但 py3不需要进行这个设置
    reload(sys) ;    sys.setdefaultencoding('utf-8')
    #2 组合落单词，使为有意义新词
    word_flag=[[word,flag] for word,flag in cut_iter]  
    #print(word_flag)
    # 2.1 基于词性的粘滞力权重，粘滞力分为向上粘滞和向下粘滞，这里默认给出向下粘滞力权重，则向上粘滞力=10-向下粘滞力权重
    prof_weight={'a':5.2,'n':3.5,'d':6.5,'v':6.5,'vn':3.5,'ns':3.5,'nt':3.5,'nz':3.5,'an':3.5,'x':0,'q':2,'uj':4,'ul':4,'m':3,'r':6} 
    stop_words_list=[u'是',u'也',u'着',u'很',u'到',u'就',u'刚',u'才',u'蛮',u'都',u'让',u'太',u'去',u'来',u'找',u'上',u'下',u'卖',u'说',u'有',u'叫',u'起',u'要',u'使',u'令']
    special_words_list=[u'不',u'没',u'无']
    wlen=len(word_flag)
    rst=[]
    if wlen>0:
        rst.append(word_flag[0])
    loci=1;not_concat=1
    while loci<wlen:
        wf_loc=word_flag[loci]
        wf_pre=word_flag[loci-1]
        try:
            wf_suf=word_flag[loci+1]
        except:
            wf_suf=['。','x']
        if len(wf_loc[0])==1 and wf_loc[0] not in stop_words_list and wf_loc[1] not in 'ujulxepcykzgqrmd' and wf_loc[0] not in special_words_list :
            # 2.3 基于词性判断，获得计算时所需的权重值
            pre_i_w=1;suf_i_w=1;loc_i_w=1
            if wf_loc[1] in prof_weight.keys():
                loc_i_w=prof_weight.get(wf_loc[1])
            if wf_pre[1] in prof_weight.keys():
                pre_i_w=prof_weight.get(wf_pre[1])
            if wf_suf[1] in prof_weight.keys():
                suf_i_w=prof_weight.get(wf_suf[1])
            if loc_i_w in [0,1]:  
                inverse_loc_i_w=loc_i_w
            else:
                inverse_loc_i_w=10-loc_i_w
            if suf_i_w in [0,1]:  
                inverse_suf_i_w=suf_i_w
            else:
                inverse_suf_i_w=10-suf_i_w
                
            # 2.4 计算粘滞方向，并做判断 ，进行连接
            #if pre_i_w*pre_i_w*inverse_loc_i_w/math.pow(len(wf_pre[0]),2)*not_concat>=loc_i_w*loc_i_w*inverse_suf_i_w/math.pow(len(wf_suf[0]),2) :
            #print(wf_loc[0],pre_i_w*pre_i_w*inverse_loc_i_w/len(wf_pre[0])*not_concat,loc_i_w*loc_i_w*inverse_suf_i_w/len(wf_suf[0]))
            pre=pre_i_w*pre_i_w*inverse_loc_i_w/math.ceil(len(wf_pre[0])/2.0)*not_concat  
            suf=loc_i_w*loc_i_w*inverse_suf_i_w/math.ceil(len(wf_suf[0])/2.0)  
            if pre>=suf and pre>0:
                if wf_pre[1]!=wf_loc[1]:
                    del rst[-1]  # 先删掉上一个
                    rst.append([wf_pre[0]+wf_loc[0],wf_pre[1]+wf_loc[1]]) # 把连接的存入result集合
                else:
                    rst.append(wf_loc)
                loci=loci+1
                not_concat=1
            elif pre<suf and suf>0:
                if wf_suf[1]!=wf_loc[1]:
                    rst.append([wf_loc[0]+wf_suf[0],wf_loc[1]+wf_suf[1]])
                    if len(wf_suf[0])>1 :  
                        loci=loci+2
                    else:
                        loci=loci+1
                else:
                    rst.append(wf_loc)
                    loci=loci+1
                not_concat=0.2
            else:
                rst.append(wf_loc)
                loci=loci+1
                not_concat=1
        elif wf_loc[0] in special_words_list and wf_suf[1]!='x':
            rst.append([wf_loc[0]+wf_suf[0],wf_loc[1]+wf_suf[1]])
            if len(wf_suf[0])>1 :
                loci=loci+2
            else:
                loci=loci+1
            not_concat=0.2
        else:
            rst.append(wf_loc)
            loci=loci+1
            not_concat=1
    
    # 3 输出有效词 作为结果 
    rst=[(text_id,text,
          [x[0] for x in rst if re.search('[^x]',x[1])],
          [{'word':x[0],'flag':x[1]} for x in rst if re.search('[^x]',x[1])]
          )] # 面向element的flatmap  
    #print(rst)
    return rst


def word_cut(strx,concat=0):
    '''
    切分词 并 根据上下文 组合落单字，使生成可能的有效词。
    return 一个list
    '''
    reload(sys) ;    sys.setdefaultencoding('utf-8')
    # 在spark上分步测试，先不加载自定义词袋 
    jieba.load_userdict('jiebaNewWords.txt')
    #从sqlContext读的dataframe RDD是基于 Row的，所以需要判断strx类型
    if type(strx)==type(sp_row()):
        idx=strx['product_id']
        strx=strx['text_note']
    confuse_words=[u'为什么',u'什么',u'怎么']
    for cf_wd in confuse_words:
        pat=re.compile(cf_wd)
        strx=pat.sub('',strx)
    replace_words=[(u'不太',u'不'),(u'不是',u'不')]
    for rp_wd in replace_words:
        pat=re.compile(rp_wd[0])
        strx=pat.sub(rp_wd[1],strx)
    words=cut(strx,HMM=True) 
    #for x,y in words :print x,y
    if concat==1:
        words_new=concat_prop_words(cut_iter=words,text=strx,text_id=idx)
    else:
        word_flag=[[word,flag] for word,flag in words]
        words_new=[(idx,strx,
                    [x[0] for x in word_flag if re.search('[^x]',x[1])],
                    [{'word':x[0],'flag':x[1]} for x in word_flag if re.search('[^x]',x[1])]
                    )]
    return words_new



if __name__=='__main__':
    confx=SparkConf().setAppName('1_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")
    
    # 1 获取数据
    df0=data_collect(sqlContext)
    
    # 2 切词 并返回 预处理落单词组合和高关联词组合后的 有效词性的词  -- 筛选有效词性后可能返回空的切词结果列表
    doc_table=params['shared']['doc_cut_word_table']
    df1=sqlContext.createDataFrame(df0.rdd.flatMap(lambda x: word_cut(x,concat=0)),['product_id','text_note','inputwds','cut_word_flag'])
    df1.write.saveAsTable('{0}.{1}'.format(database_name,doc_table),mode='overwrite')
    df1=sqlContext.sql('select * from {0}.{1}'.format(database_name,doc_table))

    topic_table=params['shared']['topic_cut_word_table']
    df2=df1.join(df0.select('product_id','cat_id'),'product_id','inner').withColumn('inputwds_concat',F.concat_ws('_',F.col('inputwds')))
    df3=df2.groupBy('cat_id').agg(F.split(F.concat_ws('_',F.collect_set(F.col('inputwds_concat'))),'_').alias('inputwds'))
    df3.write.saveAsTable('{0}.{1}'.format(database_name,topic_table),mode='overwrite')
    #df3=sqlContext.sql('select inputwds from {0}.{1}'.format(database_name,topic_table))