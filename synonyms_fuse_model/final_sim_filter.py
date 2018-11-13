# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:35:02 2018

@author: zhangyaxu
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext

from pyspark.sql import functions  as F

#import os
import pandas as pds
import numpy as npy

from sklearn.linear_model import LogisticRegression as LR
#from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.tree import DecisionTreeClassifier as DTC

from params import params

database_name=params['shared']['database_name'] #'recommend'

def cuple_wd_is_sim(x,unrelated_check=True,rel_max_n=20,windowx=3):
    '''
    desc: first check whether vec-sim(w2v and tfidf and jaro) , if vec-sim then check relate-condition while unrelated_check is True.
    params:
    x : a Row type of spark-dataframe or a series type which is one row of pandas-dataframe  ,include columns :
        columns_new=[
        'target_word','sim_word','w2v_similarity','jaro_sim','tfidf_similarity',
        'tf_max','tf_min','df_max',
        'tfidf_plus_w2v','tfidf_plus_jaro',
        'flag_cond','avg_relevance','co_occurence_num','co_occurrence_cond',
        'p_coef_max','pf_weight_max','p_over_thr1_max',
        'context_num_coef_max','unrichness_coef_max',
        'p_coef_min','pf_weight_min',
        'p_over_thr1_min','context_num_coef_min','unrichness_coef_min'
        ]
    unrelated_check: default True
    rel_max_n: control the related-max-doc-num ,here we means  product-number that words co-occurrence
    return : 1 is-sim ,2 vec-sim-only  0 not-sim
    '''
    def cuple_wd_keep_unrelated(x,rel_max_n=rel_max_n):
        '''
        共现度系数高于4的，直接不保留.
        then :
        保留 关联度低 or 虽然关联度高，但是文本相似度>0且词性相似度>=0.5 的词 进一步核实为近义词
        （无关联的词 也不保留，观察发现，在足够的语料量tf的情况下，近义词很难出现从未共同去描述同一对象的情况）
        (完全关联的词avg=1 需要做更多额外的严格判断，因为即使文本和词性层面判断是相似了，近义词也不可能出现在所有doc中都是完全关联的。所以 同时满足 总交集文档数n=final/avg^2 * < tf/5 ，且n有阈值的限制 ，这样才能保留完全关联的词)
        '''
        keep_sim=0
        avg_rel_thr=1.0/windowx**(windowx/2.0)
        #final_rel_thr=1.0 #and 0<x.final_relevance_coef<final_rel_thr
        
        # if x.flag_flag!=None:
        #     flag_cond=len(set(x.flag_flag).intersection(set(x.target_flag)))*1.0/max(len(x.flag_flag),len(x.target_flag)) # flag 相似度
        #     co_occurence_num=x.final_relevance_coef*1.0/x.avg_relevance**2 # 共现doc数
        #     co_occurrence_cond= co_occurence_num/(min(x.tgt_tf,x.sim_tf)/5.0)  # 共现度系数
        # else:
        #     flag_cond=0.0
        #     co_occurence_num=0
        #     co_occurrence_cond=0.0
        # 存在None 全无关联 ，而python 把None 对应到0 ，所以条件上需要注意>0
        co_occurence_cond_thr=4.0
        jaro_sim_thr=0.3
        flag_cond_thr=0.5
        if  x.co_occurrence_cond<co_occurence_cond_thr:
            if 0<x.avg_relevance<avg_rel_thr  :
                #and 0<x.final_relevance_coef<final_rel_thr
                #低关联度 直接保留 
                keep_sim=1
            else:
                if x.flag_cond!=0  and x.jaro_sim>jaro_sim_thr and x.flag_cond >= flag_cond_thr :
                    #高关联度的一般额外判断 
                    if x.avg_relevance<0.9:
                        keep_sim=1
                    elif x.co_occurrence_cond<1.0  and x.co_occurence_num <rel_max_n :
                        #完全关联时 需要额外的严格判断 
                        keep_sim=1    
        return keep_sim
    
    tfidf_thr2=0.7
    w2v_plus_tfidf_thr=1.5
    jaro_thr=0.8
    tfidf_plus_jaro_thr=0.95
    # if :  tfidf>=0.7  and tfidf_plus_w2v>=1.5
    # elif: tfidf>=0.7 and tfidf_plus_w2v<1.5 and jaro_sim>=0.8
    # elif: 0.55<=tfidf<0.7 and tfidf_plus_jaro>=0.95
    tfdf_cond=(x.tf_min<rel_max_n)&(x.df_max<=5)
    #unrichness_cond=(x.p_coef_max<0.4)|(x.p_over_thr1_min<1.0)|(x.context_num_coef_min<1.0)
    #unrichness_cond=(x.p_coef_max>0.6)|((x.p_coef_max>0.4)&(x.context_num_coef_min<1.0))
    def unrichess_cond_func(x):
        if x.p_coef_max>0.6 or x.p_over_thr1_min>1.2:
            unrichness_cond=True
        elif x.p_coef_max>0.4 and x.jaro_sim>=0.5 and x.p_coef_max+x.p_over_thr1_min>0.85:
            unrichness_cond=True
        elif x.p_coef_max>0.4 and x.jaro_sim<0.5 and x.p_coef_max+x.p_over_thr1_min>0.7:
            unrichness_cond=True
        elif x.p_coef_max>0.3 and x.p_over_thr1_min>1.2:
            unrichness_cond=True
        else:
            unrichness_cond=False
        return  unrichness_cond
    unrichness_cond=unrichess_cond_func(x)
    if x.tfidf_similarity>=tfidf_thr2 and x.tfidf_plus_w2v>=w2v_plus_tfidf_thr :
        if unrelated_check==True and cuple_wd_keep_unrelated(x)==1 and x.tf_max>=rel_max_n and tfdf_cond==False and unrichness_cond==False:
            return 1 
        else:
            return 2
    elif x.tfidf_similarity>=tfidf_thr2 and x.tfidf_plus_w2v<w2v_plus_tfidf_thr and x.jaro_sim>=jaro_thr: 
        if unrelated_check==True and cuple_wd_keep_unrelated(x)==1 and x.tf_max>=rel_max_n and tfdf_cond==False and unrichness_cond==False:
            return 1 
        else:
            return 2
    elif x.tfidf_similarity<tfidf_thr2 and x.tfidf_plus_jaro>=tfidf_plus_jaro_thr :
        if unrelated_check==True and cuple_wd_keep_unrelated(x)==1 and x.tf_max>=rel_max_n and tfdf_cond==False and unrichness_cond==False:
            return 1
        else:
            return 2
    else:
        return 0

def final_sim_filter(sqlContext,sim_join,windowx=3):
    """
     v1.0 模型整合策略，主要是通过规则
    :param sqlContext:
    :param sim_join:
    :param windowx:
    :return:
    """
    # 3 策略整合判断筛选
    ori_len = len(sim_join.columns)  #
    sqlContext.registerDataFrameAsTable(sim_join, 'sim_join')
    rel_max_n_thr = sqlContext.sql(' select percentile(tf_max,0.95) as tf95 from sim_join ').collect()[0][0] / 5 * 0.1 # tf词频 阈值
    #
    sim_recog = sqlContext.createDataFrame(sim_join.rdd.map(lambda x: (x[0:ori_len]) + (
    cuple_wd_is_sim(x=x, unrelated_check=True, rel_max_n=rel_max_n_thr, windowx=windowx),)),
                                           sim_join.columns + ['is_sim'])
    sqlContext.sql('drop table if exists {0}.word_semantic_similarity'.format(database_name))
    sim_recog.filter(F.col('is_sim') >= 0).write.saveAsTable('{0}.word_semantic_similarity'.format(database_name), mode='overwrite')


def split_data_random_by_index(dfx,part_num=2,split_type='weight',weights=None):
    """
    随机性的分组, 均匀分组式的随机选择 或者 按比例权重分组式的随机选择 或者 组数也是随机式的随机选择,
    切分训练集 测试集， 或者K-折交叉验证时需要这个函数
    :param dfx : a pandas dataframe
    :param split_type:  str , [uniform,weight] ,
    :param part_num: int, the num to split
    :param weights: if the split_type is 'weight' , weights should be a list or array that give each part's records-num-ratio ,so each weight should less than 1.0
            if sum(weights) <= 100%  : no-back dividing
            if sum(weights) >100% : backing dividing
            and len(weights) must equal to part_num
    :return:  list<array<int>> type , each split part's index (this is iloc type index ,applied should be like  df.iloc[...] ）
    """
    x=dfx.iloc[:,0].copy() # 节约空间
    random_index=npy.array(range(0,x.shape[0]))
    #random_index=dfx.index.values   # if dfx has multi-index, this will cause error in the next scripts
    npy.random.shuffle(random_index)  # 打散待筛选的index位置
    # 按照 split type 进行不同方式的分割
    #weights=[0.2,0.4,0.3]
    if weights==None and split_type=='weight':
        weights=[npy.random.rand()/2.0 for ii in range(part_num)] # 随机分配比例
    if split_type=='uniform':
        rm_records=int(random_index.shape[0]*1.0%part_num)
        random_index=random_index[rm_records:]
        iloc_index_tem=npy.split(random_index,part_num)  # list<array<int>> type
        iloc_index=[]
        for ii in range(part_num):
            if ii==0:
                iloc_index.append(npy.concatenate([random_index[0:rm_records],iloc_index_tem[ii]]))
            else:
                iloc_index.append(iloc_index_tem[ii])
    elif split_type=='weight':
        if sum(weights)<=1.0 and len(weights)==part_num:
            records_num=npy.array(weights)*random_index.shape[0]
            records_num_end=records_num.astype(npy.int).cumsum()  # astype int  may cause cut one record,this is  ok
            records_num_start=npy.array([0]+records_num_end.tolist()[0:len(weights)-1])
            records_range=npy.concatenate((records_num_start.reshape(-1,1),records_num_end.reshape(-1,1)),axis=1)
            iloc_index=[random_index[rangex[0]:rangex[1]] for rangex in records_range]
        elif sum(weights)>1.0 and len(weights)==part_num:
            records_num = npy.array(weights) * random_index.shape[0]
            records_num=records_num.astype(npy.int)
            iloc_index=[]
            for numx in records_num:
                npy.random.shuffle(random_index)
                iloc_index.append(random_index[0:numx])
        else:
            iloc_index='error: length of weights param is not equal to part_num,please check'
    else:
        iloc_index='error: split type is not support'
    return iloc_index


def label_pred_error(y,y_p):
    ynew=pds.DataFrame(npy.concatenate((y.reshape(-1,1),y_p.reshape(-1,1)),axis=1),columns=['y','y_p'])
    ynew['is_right']=ynew.apply(lambda x : 1 if x[0]==x[1] else 0,axis=1)
    acc_rate=ynew.groupby('y_p')['is_right'].agg(['sum','count']).reset_index()
    acc_rate['acc_rate']=acc_rate['sum']/acc_rate['count']
    cbk_rate=ynew.groupby('y')['is_right'].agg(['sum','count']).reset_index()
    cbk_rate['cbk_rate'] = cbk_rate['sum'] / cbk_rate['count']
    return acc_rate,cbk_rate


def k_fold_cross_fit(train_data,test_data,model_classes,model_stacking_weight,confidence=0.05,k=4,stacking_type='DT',uniform_voting=False):
    """v2.0 融合方式之一 ：k折交叉验证stacking方法"""
    lk = list(range(k))
    iloc_index=split_data_random_by_index(dfx=train_data,part_num=k,split_type='uniform',weights=None)
    def apply_voting_uniform(x):
        """
        :param x:  1-D list or array
        :return:  the most show times value in x
        """
        return max(set(list(x)),key=list(x).count)
    #
    def k_fold_single_model(model_class=LR,lk=lk,iloc_index=iloc_index,train_data=train_data,model_params={},rst_voting=False):
        """"""
        dikr=[(lkx,[lky for lky in lk if lky!=lkx]) for lkx in lk]
        mdls_all=[]
        y_p_all=[]
        for kx,kys in dikr:
            ilocx=iloc_index[kx]
            train_datax=train_data.iloc[ilocx, :].copy()
            mdls = []
            y_p=[]
            for ky in kys:
                ilocy=iloc_index[ky]
                train_datay = train_data.iloc[ilocy,:].copy()
                mdl=model_class()
                if len(model_params)>0:
                    mdl.set_params(**model_params)
                mdl.fit(train_datay.iloc[:,1:],train_datay.iloc[:,0])
                mdls.append(mdl)
                y_p.append(mdl.predict(train_datax.iloc[:,1:]).reshape(-1,1))
            if rst_voting==True:
                y_p_all.append((kx,npy.apply_along_axis(func1d=apply_voting_uniform,axis=1,arr=npy.concatenate(y_p,axis=1))))
            else:
                y_p_all.append((kx, npy.concatenate(y_p, axis=1)))
            mdls_all.append((kx,mdls))
        return  mdls_all,y_p_all
    #
    #  k-fold result of each model
    kfold_yp=[]
    kfold_mdls =[]
    for model_class ,model_params in model_classes:
        mdls_all,y_p_all=k_fold_single_model(model_class=model_class, lk=lk, iloc_index=iloc_index, train_data=train_data,model_params=model_params,rst_voting=uniform_voting)
        kfold_yp.append(dict(y_p_all))  # several models k-fold result
        kfold_mdls.append(dict(mdls_all))  # several models k-fold-train-mdl
    #
    # concat k-fold result of each model
    yp_stacking=[]
    for yp_dictx in kfold_yp :
        # select one model-class
        kfold_yp_concat=[]
        for  kx in lk:
            # concat the predict array y_p of each fold
            kfold_yp_concat.append(yp_dictx.get(kx))
        yp_stacking.append(npy.concatenate(kfold_yp_concat))
    #
    # stacking
    yp_stacking = npy.concatenate(yp_stacking,axis=1)
    #acc, cbk = label_pred_error(y=train_data.iloc[npy.concatenate(iloc_index), 0].values, y_p=yp_stacking.T[5]);print(acc, '\n\n', cbk)
    #
    ## 方法1 ：使用weight_voting ,弃用，不好设计权重
    if uniform_voting==False:
        model_stacking_weight=[x for x in model_stacking_weight for ii in range(k - 1)]
    def apply_voting_weight(x,weights=model_stacking_weight,cf_val=confidence):
        """"""
        stacking_decay=1.0/(len(x)-1)*(len(x)/2.0)
        tem=pds.DataFrame(npy.concatenate([x.reshape(-1,1),npy.array(weights).reshape(-1,1)],axis=1),columns=['v','w'])
        max_weights=tem.groupby(by='v')['w'].sum().reset_index().sort_values(by='w',ascending=False).values.tolist()[0]  # [val,weight_sum]
        if max_weights[1]*stacking_decay>=1.0-cf_val:
            # 超过置信度，进行分类
            return max_weights[0]
        else:
            # 否则无法分类
            return  None
    ##  方法2： 包装一层 DTC
    stack_mdl=DTC(max_depth=3)
    stack_mdl.fit(X=yp_stacking,y=train_data.iloc[npy.concatenate(iloc_index),0])
    if stacking_type=='weight_voting':
        train_yp = npy.apply_along_axis(apply_voting_weight, axis=1,arr=yp_stacking)  # 本来想为每个model加权重，继续使用 voting的方法，但是考虑到不超过置信度会导致无法分类,所以考虑另一种方法
    else:
        train_yp=stack_mdl.predict(yp_stacking)
    #acc, cbk = label_pred_error(y=train_data.iloc[npy.concatenate(iloc_index), 0].values, y_p=train_yp); print(acc, '\n\n', cbk)
    #
    #stacking_type='DT'
    def predict_custom(test_data,kfold_mdls=kfold_mdls,stack_mdl=stack_mdl,k=k,stacking_type=stacking_type,uniform_voting=uniform_voting):
        """"""
        lk = list(range(k))
        dikr = [(lkx, [lky for lky in lk if lky != lkx]) for lkx in lk]
        iloc_index = split_data_random_by_index(dfx=test_data, part_num=k, split_type='uniform', weights=None)
        kfold_yp=[]
        for kflod_mdl in kfold_mdls:
            y_p_all = []
            for kx, kys in dikr:
                ilocx = iloc_index[kx]
                test_datax = test_data.iloc[ilocx, :].copy()
                kfold_mdl_sub=kflod_mdl.get(kx) # 获得当前折区的模型
                mdl_index=0
                y_p = []
                for ky in kys:
                    ilocy=iloc_index[ky]
                    test_datay = test_data.iloc[ilocy,:].copy()
                    y_p.append(kfold_mdl_sub[mdl_index].predict(test_datay.iloc[:,1:]).reshape(-1,1))
                    mdl_index=mdl_index+1
                if uniform_voting == True:
                    y_p_all.append((kx, npy.apply_along_axis(func1d=apply_voting_uniform, axis=1,arr=npy.concatenate(y_p, axis=1))))
                else:
                    y_p_all.append((kx, npy.concatenate(y_p, axis=1)))
            kfold_yp.append(dict(y_p_all))
        yp_stacking = []
        for yp_dictx in kfold_yp:
            # select one model-class
            kfold_yp_concat = []
            for kx in lk:
                # concat the predict array y_p of each fold
                kfold_yp_concat.append(yp_dictx.get(kx))
            yp_stacking.append(npy.concatenate(kfold_yp_concat))
        yp_stacking = npy.concatenate(yp_stacking,axis=1)
        if stacking_type=='weight_voting':
            test_yp=npy.apply_along_axis(apply_voting_weight, axis=1, arr=yp_stacking)
        elif stacking_type=='DT':
            test_yp=stack_mdl.predict(yp_stacking)
        else:
            test_yp='error: stacking type is not support'
        return  test_yp,iloc_index
    test_yp,iloc_index=predict_custom(test_data=test_data.copy(), kfold_mdls=kfold_mdls, stack_mdl=stack_mdl, k=k,stacking_type=stacking_type,uniform_voting=uniform_voting)
    #acc,cbk=label_pred_error(y=test_data.iloc[npy.concatenate(iloc_index),0].values,y_p=test_yp)
    #print(acc,'\n\n',cbk)
    return (train_yp,test_yp,iloc_index)


def final_sim_pred(sqlContext,database_name,windowx=3,kfold_use=False,version='v1'):
    """v2.0版本，加入 unrichness特征并使用模型预测"""
    mix_model_data=sqlContext.sql('select * from {0}.synonyms_mix_model_data'.format(database_name)).toPandas()
    label_data=sqlContext.sql('select * from {0}.synonyms_label_sample'.format(database_name)).toPandas()
    #
    ## 1) join feature 和 label
    df0=pds.merge(mix_model_data,label_data,how='left',on=['target_word','sim_word'])
    df0.set_index(['target_word','sim_word'],inplace=True)
    feature_col=df0.columns[0:-1].tolist()
    label_col=df0.columns[-1]
    #df0.head(100)
    #
    ## 2)  基于规则融合的结果 rule_pred
    rel_max_n_thr =df0.tf_max.describe(percentiles=[0.95])['95%']/5 * 0.1 # tf词频 阈值
    df0['rule_pred']=df0.apply(lambda  x :cuple_wd_is_sim(x=x, unrelated_check=True, rel_max_n=rel_max_n_thr, windowx=windowx),axis=1)
    #
    ## 3)  基于model融合预测的结果 mdl_pred
    if version=='v2':
        ### 划分 train & test data
        iloc_index=split_data_random_by_index(dfx=df0.loc[df0.is_sim.notna()==True,:].copy(),part_num=2,split_type='weight',weights=[0.7,0.3])
        train_data=df0.loc[df0.is_sim.notna()==True,:].iloc[iloc_index[0],:][[label_col]+feature_col].copy() # label放到首列
        test_data=df0.loc[df0.is_sim.notna()==True,:].iloc[iloc_index[1],:][[label_col]+feature_col].copy() # label放到首列
        ### 单模型
        mdl=GBDT( learning_rate=0.1, n_estimators=50, max_depth=3) # 87% ,88%
        mdl.fit(train_data[feature_col],train_data[label_col])
        df0['mdl_pred']=mdl.predict(df0[feature_col])
        ### 多模型k-fold stacking  --when there are a few label samples , not recommend
        #kfold_use=False # defalut not run
        if kfold_use:
            k=4
            model_classes = [(LR, {'penalty': 'l2'}),
                         (GBDT, {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 6})]
            model_stacking_weight = [0.7, 0.7]
            confidence = 0.05
            uniform_voting = False
            kfold_train_pred,kfold_test_pred,iloc_index=k_fold_cross_fit(train_data=train_data,test_data=df0[[label_col]+feature_col],model_classes=model_classes,model_stacking_weight=model_stacking_weight,confidence=confidence,k=k,stacking_type='DT',uniform_voting=uniform_voting)
            df0.iloc[iloc_index,'kfold_pred']=kfold_test_pred
        else:
            df0['kfold_pred']=''
    else:
        df0['mdl_pred']=''
        df0['kfold_pred']=''
    #
    # 4) storage
    try:
        df0['mdl_pred']=df0.mdl_pred.astype('int')
    except:
        print('')
    try:
        df0['kfold_pred'] = df0.kfold_pred.astype('int')
    except:
        print('')
    sim_recog=sqlContext.createDataFrame(df0.reset_index())
    sqlContext.sql('drop table if exists {0}.word_semantic_similarity'.format(database_name))
    sim_recog.write.saveAsTable('{0}.word_semantic_similarity'.format(database_name), mode='overwrite')


if __name__=='__main__':
    confx=SparkConf().setAppName('7_word_semantic_similarity_on_prod_name')
    sc=SparkContext(conf=confx)
    sqlContext=HiveContext(sc)
    sc.setLogLevel("WARN")

    # 3 模型融合predict
    mix_model_data_df=sqlContext.sql('select * from {0}.synonyms_mix_model_data'.format(database_name))

    # 3.1  v1.0 版本 rule_pred ，模型的整合是基于筛选策略的，虽然准确率 可达到80%,但随着需要融合的模型增多，这个策略需要控制的参数过多,不具备拓展性
    #final_sim_filter(sqlContext, sim_join=mix_model_data_df, windowx=3)

    # 3.2  v2.0 版本 model_pred ： 新增 unrichness子模型 ，并使用GBDT进行预测，准确率可达88%,且即使后面模型继续改进并新增子模型，model_pred依然具有拓展性
    windowx=params['shared']['windowx']
    kfold_use=params['final_sim']['kfold_use']
    version=params['final_sim']['version']
    mix_model_data = mix_model_data_df.toPandas()
    final_sim_pred(sqlContext, database_name=database_name, windowx=windowx,kfold_use=kfold_use,version=version) # 注意，feature和label数据已经提前存入指定的数据库的表中，所以v2.0版本不需要在调用方法时传入数据

    # label data is load from outfile
    # label_table_create="""
    # create table {0}.synonyms_label_sample(
    # target_word string ,
    # sim_word string,
    # is_sim int
    # )row format delimited fields terminated by '\t'
    # """.format(database_name)

    # sqlContext.sql(label_table_create).collect()
    # sqlContext.sql("load data inpath '/hiveweb/recommend.db/synonyms_label_data.txt' overwrite into table {0}.synonyms_label_sample".format(database_name))


    # read data
    # filepath1='F:\\张雅旭工作文件\\1_搜索推荐相关\\4_mSearch优化之近义词识别'
    # feature_data='\\聚美平台语料的近义特征集.txt'
    # f1=open(filepath1+feature_data,encoding='utf-8')
    # mix_model_data=pds.read_table(f1,sep='\t',header=0)
    # f1.close()
    #
    # filepath2='E:\\spyder_workfiles\\data_update_summary\\word_synonyms_similar'
    # label_data='\\synonyms_label_data.txt'
    # f2=open(label_data,encoding='utf-8')
    # label_data=pds.read_table(f2,sep='\t',header=0)
    # f2.close()

    #mdl=LR(penalty='l2')  # 准确率 73% 召回率 58%
    #mdl=SVC()  # 准确率 99% 召回率 40%
    #mdl=DTC(max_depth=5) # 准确率 72% 召回率 72%
    #mdl=GBDT( learning_rate=0.1, n_estimators=50, max_depth=3) # 87% ,88%
    #mdl.fit(train_data[feature_col],train_data[label_col])
    # y_p=mdl.predict(test_data[feature_col])
    # acc,cbk=label_pred_error(y=test_data[label_col].values,y_p=y_p)
    # print(acc,'\n\n',cbk)
    #df0['mdl_pred']=mdl.predict(df0[feature_col])
    #df0.reset_index().to_csv(filepath1+"\\rule_pred_v2.csv", index=False)


















