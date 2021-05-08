# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 12:41:22 2021

@author: yxzha

seq2seq的序列预测的数据预处理 ： 
将文本映射到数值矩阵: N × T × K
N是样本句子数 ，encoder decoder 是一样的，因为是一一对应获取的
T是最大的句子长度(T1 for encoder,T2 for decoder) ，与训练时的时刻t对应
K是字的字典的总长度(K1 for encoder, K2 for decoder),指字的one-hot的维度


"""
import numpy as npy
import pandas as pds

def analog_data():
    x_chrs='abcdefghigklmnopqrstuvwxyz'
    y_chrs='ABCDEFGHIGKLMNOPQRSTUVWXYZ'
    # 0123456789 -> 1234567890 逐个对应
    
    t1=npy.random.randint(3,10)
    xright_num=npy.random.randint(10)
    last_xright_num=0
    xleft_chr=x_chrs[npy.random.randint(26)]
    input_x_chrs=[]
    output_y_chrs=[]
    for ti in range(t1):
        input_x_chrs.append(xleft_chr+str(xright_num))
        # y->map x
        yleft_chr=xleft_chr.upper()
        if xright_num>5:
            output_y_chrs.extend([yleft_chr+str(xright_num+1),yleft_chr+str(xright_num+2)])
        else:
            output_y_chrs.extend([yleft_chr+str(xright_num+1)])
        #x2 -> map x1
        sumnum=last_xright_num+xright_num
        last_xright_num=xright_num
        xright_num=sumnum%10 
        next_x_chr_idx=( x_chrs.index(xleft_chr)+(xright_num%3) )%26
        xleft_chr=x_chrs[next_x_chr_idx]
    return input_x_chrs,output_y_chrs

def analog_datas(N=1000):
    #N=1000
    datas=[]
    for i in range(N):
        input_x_chrs,output_y_chrs=analog_data()
        datas.append([input_x_chrs,output_y_chrs])
    #print(input_x_chrs,output_y_chrs)
    return datas

def data_prepare_processing(datas=None,stop_token='\n',start_token='\t',N=1000):
    if datas==None:
        datas=analog_datas(N=N)
        print('no data -> starting analog...')
    ## 1 分开 x_data,y_data 
    xdata=[] # for encoder input-data
    ydata=[] # for decoder input-data and output-data
    x_wds=set([])
    y_wds=set([start_token,stop_token])
    T1=0 # encoder input-data-sentence max-time(=max word-num)
    T2=0 # decoder input/output-data-sentence max-time(max word-num)
    for xdt,ydt in datas:
        xdata.append(xdt)
        ydata.append([start_token]+ydt+[stop_token]) # decoder 使用的y 需要加入默认的起始和结尾字符
        x_wds=set(x_wds).union(set(xdt))
        y_wds=set(y_wds).union(set(ydt))
        T1=max(T1,len(xdt))
        T2=max(T2,len(ydt)+2) #加了 \t \n
    ## 2 字典 ：正向  word as key ,index as value ; 反向 index as key ,word as value
    dict_xtoken_index=dict(pds.DataFrame(sorted(x_wds),columns=['tk']).reset_index()[['tk','index']].values.tolist())
    reverse_dict_xtoken_index=dict(pds.DataFrame(sorted(x_wds),columns=['tk']).reset_index()[['index','tk']].values.tolist())
    
    dict_ytoken_index=dict(pds.DataFrame(sorted(y_wds),columns=['tk']).reset_index()[['tk','index']].values.tolist())
    reverse_dict_ytoken_index=dict(pds.DataFrame(sorted(y_wds),columns=['tk']).reset_index()[['index','tk']].values.tolist())
    
    ## 3 将token映射位one-hot向量，数据集数值化 N × T× K 用来代入模型训练
    N=len(xdata) # 不用区分N1 N2 ,因为N1=N2，即input-data 和target-data的样本量一一对应，是相等的。
    K1=len(dict_xtoken_index) # K1 encoder-data one-hot(wd)的长度
    K2=len(dict_ytoken_index) # K2 decoder-data one-hot(wd)的长度
    ###一共需要三个数据，encoder的input-data ，decoder的input-data 和 output-data
    encoder_input_data=npy.zeros((N,T1,K1),dtype='float32')
    decoder_input_data=npy.zeros((N,T2,K2),dtype='float32')
    decoder_output_data=npy.zeros((N,T2,K2),dtype='float32')
    for ni ,(xsent,ysent) in enumerate(zip(xdata,ydata)):
        for t1,wdx in enumerate(xsent):
            encoder_input_data[ni,t1,dict_xtoken_index[wdx]]=1.0
        for t2,wdy in enumerate(ysent):
            decoder_input_data[ni,t2,dict_ytoken_index[wdy]]=1.0
            if t2>0:
                decoder_output_data[ni,t2-1,dict_ytoken_index[wdy]]=1.0
    # 3 整合
    params={'N':N,
            'T1':T1,
            'T2':T2,
            'K1':K1,
            'K2':K2}
    token_dicts={'x': dict_xtoken_index,
                 'x_revrs': reverse_dict_xtoken_index,
                 'y': dict_ytoken_index,
                 'y_revrs': reverse_dict_ytoken_index}
    fit_data={'encoder_input_data':encoder_input_data,
              'decoder_input_data':decoder_input_data,
              'decoder_output_data':decoder_output_data}
    
    return fit_data, params, token_dicts


    
if __name__=='__main__':
    # 省略外部数据集，先自建一个方便使用
    N=1000
    datas=[]
    for i in range(N):
        input_x_chrs,output_y_chrs=analog_data()
        datas.append([input_x_chrs,output_y_chrs])
    print(input_x_chrs,output_y_chrs)
    
    
    # 预处理,文本转化为数值 : data-samples N-dim -> a sample-sentence T-dim -> one-hot(word) K-dim  =>N × T × K
    #stop_token='\n';start_token='\t'
    fit_datas, params, token_dicts=data_prepare_processing(datas,stop_token='\n',start_token='\t')
    fit_datas.keys()
    encoder_input_data=fit_datas['encoder_input_data']
    decoder_input_data=fit_datas['decoder_input_data']
    decoder_output_data=fit_datas['decoder_output_data']
    N,T1,T2,K1,K2=params['N'],params['T1'],params['T2'],params['K1'],params['K2']
    dictx,dictx_revrs=token_dicts['x'],token_dicts['x_revrs']
    dicty,dicty_revrs=token_dicts['y'],token_dicts['y_revrs']


        
