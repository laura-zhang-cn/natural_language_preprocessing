# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:00:29 2018

@author: zhangyaxu

密度收敛可达的聚类算法：
类的延伸不是发散的，而是每次密度可达(相近)的新的点的数量，都要较上一次发生收敛，若出现发散或 连续无法收敛达到阈值次数的情况，则剔除发散/无法收敛的点后，存余的点生成新类 。

区分DBSCAN：
DBSACN是类的延伸应该是发散的，若发散的数量达不到阈值，则生成新类 。
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext
from params import params
#from pyspark.sql import functions  as F

#import numpy as npy

## 方案2 ，类延伸过程中引入收敛速度和延伸的深度
def one_to_all_sim2(tw,sim_term,max_deep=5,equal_convergence_thr=2):
    '''
    基于收敛速度的延伸，当类延伸过程中，越到后期，类应该延伸越慢，即触发收敛，若类的收敛速度出现扩大，发散，则认为触发到了收敛的底部，则停止聚类。
    遂加入 收敛速度 条件 ： 这里约定 每次类拓展出的新词的数量 ，定义为基于语义相似的文本聚类的类的收敛速度。
    tw ：某个初始中心词
    max_deep: 最大深度，指拓展的最大次数
    equal_convergence_thr：连续不收敛的次数
    '''
    current_tw=[tw]
    new_tw=current_tw
    lenn=100;lenc=1
    convergence0=lenn-lenc  # 由于 基于相似词 进行聚类时，类的元素个数不会减少，所以 收敛速度 convergence最小为0 即类不再扩张，完全收敛
    deepx=0 # 树深度
    equal_convergence=0  # 持续 相等收敛速度的次数，即 认为收敛到达底部，避免永远无法真实收敛的情况，及时停止。
    while convergence0>0 and deepx<=max_deep and equal_convergence<equal_convergence_thr:
        #condx=(sim_term.target_word.isin(current_tw))&(sim_term.is_cluster==0)
        condx=sim_term.target_word.isin(current_tw)
        sim_words=sim_term.loc[condx,'sim_word'].values.tolist()   # 获得相似词
        diff_tw=list(set(new_tw+sim_words).difference(set(new_tw)))  # 差集
        new_tw=list(set(new_tw+sim_words))  
        convergence1=len(new_tw)-len(current_tw)  # 收敛速度
        if convergence1==convergence0:
            equal_convergence=equal_convergence+1
        if convergence1<=convergence0  and equal_convergence<equal_convergence_thr:
            current_tw=new_tw
            deepx=deepx+1
            convergence0=convergence1
        else :
            #vercongence1>convergence0  or equal_convergence>=2
            #若开始发散或达到收敛底部，寻找发散源
            source_tw_back1=sim_term.loc[(sim_term.target_word.isin(current_tw))&(sim_term.sim_word.isin(diff_tw)),['target_word','sim_word']].groupby('target_word')['sim_word'].count().reset_index()
            source_tw=source_tw_back1['target_word'].values.tolist() # 倒数第一层的发散源
            #tw=source_tw_df.sort_values(by='sim_word',ascending=False).values.tolist()[0]  
            # 追溯倒数第二层的发散源
            source_tw_back2=sim_term.loc[(sim_term.target_word.isin(set(current_tw).difference(set(source_tw))))&(sim_term.sim_word.isin(source_tw)),['target_word','sim_word']].groupby('target_word')['sim_word'].count().reset_index()
            if len(source_tw_back2['target_word'].values)==1 and len(source_tw)>1:
                # 追溯倒数第2层，若 倒数第一层 有多个发散源，且同归属于某个倒数第2层的term_x 且此term_x的拓展term_y中，发散term_y1多于收敛term_y2,则 倒数第2层的term_x也不能归纳到当前类中，需要剔除
                term_y_num=sim_term.loc[sim_term.target_word.isin(source_tw_back2['target_word'].values),['target_word','sim_word']].shape[0] # term_x拓展的term_y总数量
                term_y1=len(source_tw) # 发散的term_y1数量
                term_y2=term_y_num-term_y1  # 收敛的term_y2数量
                if term_y1>term_y2:
                    source_tw.extend(source_tw_back2['target_word'].values.tolist())
            current_tw=list(set(current_tw).difference(set(source_tw)))
            break
    return current_tw



if __name__=='__main__':
    
    confx=SparkConf().setAppName('7_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")
    
    
    # 0 数据准备
    database_name=params['shared']['database_name']
    sim_col='mdl_pred'
    sim_term=sqlContext.sql('select * from {0}.word_semantic_similarity where {1}=1'.format(database_name,sim_col)).toPandas()
    target_term=sim_term.groupby('target_word').agg({'sim_word':'count','tf_min':'max'}).rename(columns={'sim_word':'sim_word_num'}).sort_values(by=["sim_word_num","tf_min"], ascending=[0, 0])
    target_term['is_cluster']=0
    sim_term['is_cluster']=0
    # 1 开始训练
    tws=target_term.index.values
    for tw in tws:
        if target_term.loc[tw,'is_cluster']==0:
            tw_cluster_sim=one_to_all_sim2(tw,sim_term=sim_term,max_deep=5,equal_convergence_thr=2)
            target_term.loc[set(tw_cluster_sim).intersection(set(tws)),'is_cluster']=1
            target_term.loc[tw,'sim_words_all']=','.join(tw_cluster_sim)
            sim_term.loc[sim_term.sim_word.isin(tw_cluster_sim),'is_cluster']=1
        else:
            continue

    clusters=target_term.loc[target_term.sim_words_all.notna(),:].reset_index().rename(columns={'target_word':'center_word'})
    #clusters.head()
    clusters.sort_values(by='tf_max',ascending=False,inplace=True)
    sqlContext.createDataFrame(clusters.loc[:,['center_word','sim_words_all']]).write.saveAsTable('{0}.word_semantic_cluster'.format(database_name),mode='overwrite')


    ## check
    #clusters.loc[clusters.center_word==u'洁面',:]
    #clusters.loc[clusters.center_word==u'净白',:]
    #clusters.loc[clusters.center_word==u'水润',:]
    #clusters.loc[clusters.center_word==u'洁面',:]
    #clusters.loc[clusters.center_word==u'透白',:]
    #clusters.loc[clusters.center_word.isin([u'眉膏',u'眉粉',u'眉形']),:]
    #

    ## 方案1 ： 类具有无限延伸性，出来的效果与目标存在明显差异，弃用。
    #def one_to_all_sim(tw,sim_term=sim_term):
    #    '''
    #    无限制性的类延伸，直到类无法延伸（有问题）
    #    '''
    #    current_tw=[tw]
    #    new_tw=current_tw
    #    lenn=2
    #    lenc=1
    #    while lenc<lenn:
    #        new_tw2=new_tw+sim_term.loc[sim_term.target_word.isin(current_tw),'sim_word'].values.tolist()
    #        new_tw=list(set(new_tw2))
    #        new_tw.sort()
    #        lenn=len(new_tw)
    #        lenc=len(current_tw)
    #        current_tw=new_tw
    #    return ','.join(current_tw)
    #
    #
    #target_term.reset_index(inplace=True)
    #target_term['sim_words_all']=target_term.apply(lambda x :one_to_all_sim(tw=x[0],sim_term=sim_term),axis=1)
    #
    #clusters_id=target_term.groupby('sim_words_all')['tgt_tf'].idxmax().values.tolist()
    #clusters=target_term.loc[clusters_id,:].sort_values(by='tgt_tf',ascending=False).reset_index(drop=True).rename(columns={'target_word':'center_word'})
    #clusters.head()
    #sqlContext.createDataFrame(clusters.loc[:,['center_word','sim_words_all']]).write.saveAsTable('recommend.word_semantic_cluster',mode='overwrite')
    #


















