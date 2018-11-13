# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:08:39 2018

@author: zhangyaxu
"""


from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

from pyspark.sql import functions  as F
from pyspark.sql import Window

import pandas as pds
import numpy as npy

from datetime import datetime as dtm
from params import  params
database_name=params['shared']['database_name']

def tfidf_similarty(tk2id,term_tfidf,topn=50,sim_min=0.0):
    '''
    全部在python内存中运行，效率低，且会溢出，遂摒弃了这个方案。
    
    获取tk2id中所以term key的相似term  ,
    tk2id ： dict-type 获取的词与词id
    term_tfidf：pandas-dataframe-type, 词的tfidf-vector ; index 是词的id, 一行的values就是该词的tfidf-vector
    topn ： 筛选topn个， None表示不筛选（不推荐，数据多时 容易内存溢出）
    sim_min ： 保留相似度>=sim_min的结果。
    return 相似term-dataframe
    '''
    def term_tfidf_similarity(term_name,tk_id,term_tfidf,topn=50,sim_min=0.0):
        '''
        term_name: make sure it is unicode-chinese if python2 . such as  u'化妆水'
        tk2id : term and id relation ,dict-type
        term_tfidf : term的tfidf向量表达，一行是一个term在各doc的tfidf权重，所以一行就是一个term的tfidf向量，且term_id是index
        '''
        def cosine(x,y):
            return npy.dot(x.reshape(1,-1),y.T)/(npy.linalg.norm(x)*npy.linalg.norm(y,axis=1))
        
        term_id=tk_id.loc[tk_id.term_name==term_name,'term_id'].values[0]
        x=term_tfidf.loc[term_tfidf.index==term_id,:].values  # shape = (1,n)
        y=term_tfidf.loc[term_tfidf.index!=term_id,:].values  # shape = (m,n)
        condx=x.mean()==0.0
        condy=max(y.mean(axis=1)==0)==True
        other_termid=term_tfidf.loc[term_tfidf.index!=term_id,:].index.values 
        sim=cosine(x,y) # 归一化， -1==0 表示完全相反的向量  ，0==>0.5 要么是通用向量，跟谁都有点关系，也就是跟谁都没明显关系
        del x,y
        if condx and condy :
            sim[npy.isnan(sim)]=1.0
        else:
            sim[npy.isnan(sim)]=0.0
        sim=sim*0.5+0.5
        term_sim=pds.DataFrame(list(zip(other_termid.tolist(),sim.tolist()[0])),columns=['term_id','tfidf_sim']) 
        del sim
        term_sim['base_term']=term_name
        #term_sim=pds.merge(term_sim,tk_id,on='term_id',how='inner')[['base_term','term_name','tfidf_sim']]
        term_sim=term_sim[['base_term','term_id','tfidf_sim']]
        #print(term_sim)
        term_sim=term_sim.loc[term_sim.tfidf_sim>=sim_min,:].reset_index(drop=True)
        if topn!=None:
            term_sim=term_sim.sort_values('tfidf_sim',ascending=False).reset_index(drop=True).loc[0:topn,:]
        return term_sim
    
    # 循环获取每个term的相似term
    tk_id=pds.DataFrame(list(tk2id.items()),columns=['term_name','term_id']) # dataframe 方便操作
    #term_name=u'b'
    k=1
    for term_name in  tk2id.keys():
        print(k)
        term_sim=term_tfidf_similarity(term_name=term_name,tk_id=tk_id,term_tfidf=term_tfidf,topn=topn,sim_min=sim_min)
        if k==1:
            tfidf_sim=term_sim.copy()
        else:
            tfidf_sim=pds.concat((tfidf_sim,term_sim),axis=0)
        #print(tfidf_sim.shape)
        del term_sim
    
    return tfidf_sim


def tfidf_similarty_sparse_vector(sqlContext,sc,tk2id,tfidf_vec,sparse=True,sparse_val=-1.0,topn=50,sim_min=0.55,storage=True):
    '''
    由于此处调用的 tfidf_vec 还是离散向量直接存储的方式，在计算过程中非常占用内存，经常会导致driver崩溃，遂 也弃用这个方案。
    
    tk2id ： dict-type 获取的词与词id
    tfidf_vec : pandas-dataframe-type （m,1）, 词的tfidf-vector ; index 是词的id, 唯一一列是该词的tfidf-vector （被以list形式存储在 dataframe的一列）
    sparse ： 是否稀疏，当稀疏时，会进行去稀疏值，取非稀疏并集位的方式进行相似度计算。
    sparse_val: 当稀疏时，这个作为 待去除的稀疏值，并在非稀疏取并集时，对Na位置进行填充sparse_val， 然后进行计算。
    topn ： 筛选topn个， None表示不筛选（不推荐，数据多时 容易内存溢出）
    sim_min ： 保留相似度>=sim_min的结果。
    storage=True ： 是否存储到hive表
    return 相似term spark-dataframe 
    '''
    def vec_union_set(a,b,sparse_val=-1.0):
        '''
        获取两个稀疏向量 非稀疏部分的并集位置，最终二者分别保留并集位的值
        注意 a和b必须是等长的，不然位置无法对应
        a=npy.array([-1, -1,  0.2, 0.7, -1,  -1])
        b=npy.array([-1, 0.4, -1,   0,  0.7, -1])
        并集位在 [1,2,3,4] (任意一个向量的位置是非稀疏的值)
        则保留 
        a_u=[-1,0.2,0.7,-1] 
        b_u=[0.4,-1,0,0.7]
        return 保留向量
        '''
        pos_a=npy.array(range(len(a)))[a!=sparse_val].tolist()
        pos_b=npy.array(range(len(b)))[b!=sparse_val].tolist()
        pos_u=list(set(pos_a+pos_b)) # 去重，不在乎顺序
        a_u=a[pos_u]
        b_u=b[pos_u]
        return a_u,b_u
    
    def cosine_sparse_vector(a,b,sparse=True,sparse_val=-1.0):
        a=npy.array(a)
        b=npy.array(b)
        if sparse:
            a,b=vec_union_set(a,b,sparse_val=sparse_val)
        if a.mean()==0.0 and b.mean()==0.0:
            return 1.0
        elif a.mean()!=0 and b.mean()!=0:
            return npy.dot(a,b)/(npy.linalg.norm(a)*npy.linalg.norm(b))*0.5+0.5
        else:
            return 0.5
    
    def apply_cosine(df_vec,tfidf_vec,sparse=True,sparse_val=-1.0,sim_min=0.0,topn=30):
        #columnx=list(df_vec.columns)  # [term_id,tfidf_vec]
        vec_sim=df_vec.apply(lambda x: cosine_sparse_vector(a=tfidf_vec,b=x[0],sparse=True,sparse_val=-1.0),axis=1).loc[lambda x : x>=sim_min].sort_values(ascending=False).iloc[0:topn]  # an series
        return list(zip(vec_sim.index.values.tolist(),vec_sim.values.tolist()))
    
    ## 初始化 数据
    index_col='term_id'
    tfidf_vec.reset_index(inplace=True)  # [term_id,tfidf_vec]
    tfidf_vec_df=sqlContext.createDataFrame(tfidf_vec,['base_term_id','base_term_vec'])
    bc_tfidf_vec=sc.broadcast(tfidf_vec.set_index(index_col)) # 广播变量，rdd计算时调用
    
    ##cosine-similarity-calcularity 核心计算阶段
    term_sim1=tfidf_vec_df.rdd.map(lambda x : (x[0],apply_cosine(df_vec=bc_tfidf_vec.value,tfidf_vec=x[1],sparse=sparse,sparse_val=sparse_val,sim_min=sim_min,topn=topn))) # [(base_term_id,[(sim_term_id,sim_coef),..])]
    #y=term_sim1.top(5)  # 会 gc ，driver崩溃，跑不出来的
    ## rdd变形 并转化为 dataframe
    term_sim=term_sim1.flatMap(lambda y:[(y[0],)+(yy[0],yy[1]) for yy in y[1] if len(y[1])>0])  # 存在相似term时，则进行flatmap展开 使形式为 [(base_term_id1,sim_term_id1,sim_coef11),(base_term_id1,sim_term_id2,sim_coef12),..]
    tfidf_cosine=sqlContext.createDataFrame(term_sim,['base_term_id','sim_term_id','tfidf_similarity'])
    
    ## transform : get term name  #一些变形， join term的名称
    tk_id=sqlContext.createDataFrame(list(tk2id.items()),['term_name','term_id']) # dataframe 方便操作join
    tfidf_cosine=tfidf_cosine.join(tk_id,tfidf_cosine.sim_term_id==tk_id.term_id,'inner').drop('term_id').withColumnRenamed('term_name','sim_word')
    tfidf_cosine=tfidf_cosine.join(tk_id,tfidf_cosine.base_term_id==tk_id.term_id,'inner').drop('term_id').withColumnRenamed('term_name','target_word')
    tfidf_cosine=tfidf_cosine.select('target_word','sim_word','tfidf_similarity')
    
    # persist and storage 存储操作，action操作
    #tfidf_cosine.persist(StorageLevel(True,True,False,False,1))
    if storage:
        tfidf_cosine.write.saveAsTable('{0}.tfidf_sim_on_product_name'.format(database_name),mode='overwrite')
    
    return tfidf_cosine

def tfidf_similarty_unequal_length_vector(sqlContext,sc,tfidf_vec_df,tfidf_vec,sparse_val=0.0,topn=50,sim_min=0.55):
    '''
    正确的方案， 先借助broadcast 实现cross-join 的map_join ；rdd 计算相似度 并筛选等
    
    tk2id ： dict-type 获取的词与词id
    tfidf_vec : pandas-dataframe-type （m,1）, 词的tfidf-vector ; index 是词的id, 唯一一列是该词的tfidf-vector （被以list形式存储在 dataframe的一列）
    sparse_val: 这个作为对不等长的非稀疏向量取并集时，对Na位置进行填充sparse_val， 然后进行计算。
    topn ： 筛选topn个， None表示不筛选（不推荐，数据多时 容易内存溢出）
    sim_min ： 保留相似度>=sim_min的结果。
    return  ‘over’   # 注意 此部分相似结果 已经存储到hive表
    '''
    def vec_union_set(a,b,sparse_val=-1.0):
        '''
        获取两个向量  并集位置填充稀疏值
        a=[[0.2, 0.7],[1,3]]  ： 向量 和 位置
        b=[[0.4, 0,  0.7],[1,2,7]]
        并集位在 [1,2,3,7] (任意一个向量的位置是非稀疏的值)
        则保留 
        a_u=[ 0.2, -1.0, 0.7  , -1.0 ] 
        b_u=[ 0.4, 0.0 , -1.0 , 0.7]
        return 保留向量
        '''
        pos_a=a[1]
        pos_b=b[1]
        a=a[0]
        b=b[0]
        pos_u=list(set(pos_a+pos_b)) # 去重，不在乎顺序
        a_u=[sparse_val]*len(pos_u)
        b_u=[sparse_val]*len(pos_u)
        for pos in pos_u:
            if pos in pos_a:
                a_u[pos_u.index(pos)]=a[pos_a.index(pos)]
            if pos in pos_b:
                b_u[pos_u.index(pos)]=b[pos_b.index(pos)]
        return npy.array(a_u),npy.array(b_u)
    
    def cosine_sparse_vector(a,b,sparse_val=-1.0,min_intersect_rate=0.5):
        len_inters=len(set(a[1]).intersection(set(b[1])))
        if len_inters>0:
            r_a=len_inters*1.0/len(a[1]) # 交集占a的比例
            r_b=len_inters*1.0/len(b[1]) # 交集占b的比例
            if r_a<min_intersect_rate or r_b<min_intersect_rate:
                decay_coef=min(r_a,r_b)/min_intersect_rate
            else:
                decay_coef=1.0
            a,b=vec_union_set(a,b,sparse_val=sparse_val)
            if a.mean()==0.0 and b.mean()==0.0:
                return 1.0*decay_coef
            elif a.mean()!=0 and b.mean()!=0:
                return float(npy.dot(a,b)/(npy.linalg.norm(a)*npy.linalg.norm(b))*0.5+0.5)*decay_coef
            else:
                return 0.5*decay_coef
        else:
            return 0.0
    
    def apply_cosine(df_vec,tfidf_vec,sparse_val=-0.1,sim_min=0.0,topn=30):
        #columnx=list(df_vec.columns)  # [term_id,tfidf_vec]
        vec_sim=df_vec.apply(lambda x: cosine_sparse_vector(a=tfidf_vec,b=[x[0],x[1]],sparse_val=sparse_val),axis=1).loc[lambda x : x>=sim_min].sort_values(ascending=False).iloc[0:topn]  # an series
        return list(zip(vec_sim.index.values.tolist(),vec_sim.values.tolist()))
    
    #  sparse_val=0.0;topn=30;sim_min=0.45;storage=True
    
    # 1  broadcast 方法1  -- 效率：比不广播快，但是 比方法2 map-crossjoin方法的慢
    ## 初始化 数据
    #index_col='term_name'
    #tfidf_vec.set_index(index_col,inplace=True)
    #bc_tfidf_vec=sc.broadcast(tfidf_vec) # 广播变量，rdd计算时调用
    ###cosine-similarity-calcularity 核心计算阶段
    #term_sim1=tfidf_vec_df.limit(100).rdd.repartition(500).map(lambda x : (x[0],apply_cosine(df_vec=bc_tfidf_vec.value,tfidf_vec=[x[1],x[2]],sparse_val=sparse_val,sim_min=sim_min,topn=topn))) # [(base_term_id,[(sim_term_id,sim_coef),..])]
    #t1=dtm.now()
    #y=term_sim1.top(5)
    #print(dtm.now()-t1)
    ### rdd变形 并转化为 dataframe
    #term_sim=term_sim1.flatMap(lambda y:[(y[0],)+(yy[0],yy[1]) for yy in y[1] if len(y[1])>0])  # 存在相似term时，则进行flatmap展开 使形式为 [(base_term_id1,sim_term_id1,sim_coef11),(base_term_id1,sim_term_id2,sim_coef12),..]
    #tfidf_cosine=sqlContext.createDataFrame(term_sim,['target_word','sim_word','tfidf_similarity'])
    #  
    # persist and storage 存储操作，action操作
    #t1=dtm.now()
    #tfidf_cosine.write.saveAsTable('{0}.tfidf_sim_on_product_name'.format(database_name),mode='overwrite')
    #print(dtm.now()-t1)
    
    ## 2  broadcast cross map-join 方法 ，效率快
    # 但是由于需要 window操作获得排序后筛选，这个过程，效率会随着数据规模先提高 之后就幂式降低，比如 1000*5w 需要2.3分钟，但是 2000*5w就会卡住,
    # 所以注意分批，很 window操作的数据集 不要太大
    # sparse_val=0.0;topn=30;sim_min=0.5 
    
    #t1=dtm.now()
    #tfidf_vec_df=1
    endx=tfidf_vec_df.select(F.max('rankx')).collect()[0][0]
    #endx=10000
    rk_step=endx;rk_start=1;
    tem_type=3  # 1 limit , 2 filter, 3 broadcast map-join
    while rk_start<=endx:
        t1=dtm.now()
        #提取批次计算相似度
        if tem_type==1:
            tem1=tfidf_vec_df.limit(rk_step).selectExpr('1 as dkid','term_name','tfidf_vec','tfidf_vec_pos')
        elif tem_type==2:
            tem1=tfidf_vec_df.filter(F.col('rankx').between(rk_start,rk_start+rk_step-1)).selectExpr('1 as dkid','term_name','tfidf_vec','tfidf_vec_pos')
        elif tem_type==3:
            #tem1=tfidf_vec_df.filter(F.col('rankx').between(rk_start,rk_start+rk_step-1)).selectExpr('1 as dkid','term_name','tfidf_vec','tfidf_vec_pos').collect()
            tem1=tfidf_vec_df.filter(F.col('rankx').between(rk_start,rk_start+rk_step-1)).selectExpr('1 as dkid','term_name','tfidf_vec','tfidf_vec_pos').collect()
            bd_tem1=sc.broadcast(tem1)
        tem2=tfidf_vec_df.selectExpr('1 as dkid','term_name as term_name2','tfidf_vec as tfidf_vec2','tfidf_vec_pos as tfidf_vec_pos2')
        
        # join 
        if tem_type<3:
            # reduce join方法
            tfidf_vec_df_crossjoin=tem1.join(tem2,'dkid','inner')
            #tfidf_vec_df_crossjoin=tfidf_vec_df_crossjoin.rdd.map(lambda x: x[0:7])
        elif tem_type==3:
            # map-join方法
            tfidf_vec_df_crossjoin=tem2.rdd.repartition(400).flatMap(lambda x : map(lambda y : (x.dkid,y.term_name,y.tfidf_vec,y.tfidf_vec_pos,x.term_name2,x.tfidf_vec2,x.tfidf_vec_pos2),bd_tem1.value)) 
        #tfidf_vec_df_crossjoin.count() 
        #print(dtm.now()-t1) # 11  26s   11  
        
        # 核心计算
        if tem_type==3:
            tfidf_vec_df_sim_all=tfidf_vec_df_crossjoin.map(lambda x : (x[1],x[4],cosine_sparse_vector(a=[x[2],x[3]],b=[x[5],x[6]],sparse_val=sparse_val)))
        if tem_type<3:
            tfidf_vec_df_sim_all=tfidf_vec_df_crossjoin.rdd.repartition(400).map(lambda x : (x.term_name,x.term_name2,cosine_sparse_vector(a=[x.tfidf_vec,x.tfidf_vec_pos],b=[x.tfidf_vec2,x.tfidf_vec_pos2],sparse_val=sparse_val)))
        #tfidf_vec_df_sim_all.count(); print(dtm.now()-t1) 
        
        # sim_min筛选
        tfidf_cosine=sqlContext.createDataFrame(tfidf_vec_df_sim_all.filter(lambda x: x[2]>=sim_min),['target_word','sim_word','tfidf_similarity'])
        #tfidf_cosine.count(); print(dtm.now()-t1) 
        if rk_start==1:
            modex='overwrite'
        else:
            modex='append'
        tfidf_cosine.write.saveAsTable('{0}.tfidf_sim_all_on_product_name'.format(database_name),mode=modex)
        tfidf_cosine=sqlContext.sql('select * from {0}.tfidf_sim_all_on_product_name'.format(database_name))
        # topn选取
        windowy=Window.partitionBy('target_word').orderBy(F.desc('tfidf_similarity'))
        tfidf_cosine=tfidf_cosine.withColumn('rankx',F.row_number().over(windowy)).filter(F.col('rankx')<=topn).withColumn('rk_start',F.lit(rk_start)).drop('rankx')
        #tfidf_cosine.count() ;     print(dtm.now()-t1) 
        # 逐批写入表
        tfidf_cosine.write.saveAsTable('{0}.tfidf_sim_on_product_name'.format(database_name),mode=modex)
        rk_start=rk_start+rk_step
        print(dtm.now()-t1) 
    return tfidf_cosine

if __name__=='__main__':
    confx=SparkConf().setAppName('4_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")

    ## 4 tfidf 相似度
    # 4.1 生成tfidf 向量，并存储 ： 先在独立的py文件中运行了，以便释放内存
      
    # 4.2 计算相似度 建议这里重新开一个 py文件，这样可以释放内存
    tfidf_id_vec_df=sqlContext.sql('select * from {0}.term_id_tfidf_vec_in_cat'.format(database_name))
    tk_id=sqlContext.sql('select * from {0}.term_id_name'.format(database_name))
    
    tfidf_vec_df=tk_id.join(tfidf_id_vec_df,'term_id','inner').select('term_name','tfidf_vec','tfidf_vec_pos','rankx')
    tfidf_vec=tfidf_vec_df.toPandas()
    #import sys
    #sys.getsizeof(tfidf_vec)/1024.0/1024
    sparse_val=params['tfidf']['na_val']
    topn=params['tfidf']['topn']
    sim_min=params['tfidf']['sim_min']
    tfidf_cosine=tfidf_similarty_unequal_length_vector(sqlContext,sc,tfidf_vec_df,tfidf_vec,sparse_val=sparse_val,topn=topn,sim_min=sim_min)

    ## 计算 词的tfidf向量的余弦相似度，并支持返回topn个 。调用之前 失败的两个方案，弃用
    # ## term_id 和词的对应关系 {word:term_id,....}   #id2tk=dict(zip(tk2id.values(),tk2id.keys()))
    #tk2id=dict(tk_id.toPandas().values.tolist())
    #tfidf_sim=tfidf_similarty(tk2id=tk2id,term_tfidf=term_tfidf,topn=30,sim_min=0.55)
    #tfidf_sim=tfidf_similarty_sparse_vector(sqlContext,sc,tk2id,tfidf_vec,sparse=True,sparse_val=-1.0,topn=30,sim_min=0.55,storage=True)
    #sparse=True;sparse_val=-1.0;topn=30;sim_min=0.55;storage=True


    #x=[0.0687139273586093,0.26924664185961983,0.017090797817728192,0.1590905459402894,0.004967842258367511,0.4693181818181818,0.10206776801078096,0.028519914310767707,0.0270236216173122,0.014201848214739464,0.015545638590167903,0.0037373093726070067,0.1050733027445443,0.016825410566136744,0.026545443563624496,0.1417261306686095,0.014507085026813998,0.007311542916732695,0.005270590752859457,0.04538167787191872,0.07044361898916193,0.043913385128132316,0.0018616950413776838,0.01604549746493062,0.0016649624693130737,0.19158534869198032,0.019279047129423944,0.029598094052775126,0.009246162415775746,1.0,0.8539300224672597,0.3620851976056391,0.005222188222083615,0.20422210396399176]
    #
    #y=[0.009238269090663888,0.23071799674926613,0.6621621621621621,0.06018738162797896,0.023124254846568416,0.038383373553349906,0.00202016722843622,0.03183185782782627,0.030996748508415883,0.05571543220387598,0.36333059796885514,0.09410001098473945,0.005778302106633069,0.10292499285558142,0.02474830434424806,0.03418761569422351,0.027102919272619087,0.17662058415017018,0.07615526377206697,0.013009862809403206,0.010799756557706425,0.050093326001714476,0.0684995340167754,0.1064665414409902,0.006399587903302731,0.02998755378089431,1.0,0.8247416353717971,0.42155449201488127,0.11039032646702258]
    #
    #xp=[15,32,78,80,107,116,167,193,236,241,248,266,294,298,328,348,406,478,486,515,519,647,675,719,735,914,967,987,994,1026,1033,1040,1086,1192]
    #
    #yp=[78,80,116,167,193,241,266,298,314,348,395,406,419,439,453,486,491,515,519,719,735,743,821,827,987,994,1026,1033,1040,1192]
    #
    #
    #a=[x,xp]
    #b=[y,yp]
    #sparse_val=0.0
    #cosine_sparse_vector(a=a,b=b,sparse_val=sparse_val)
