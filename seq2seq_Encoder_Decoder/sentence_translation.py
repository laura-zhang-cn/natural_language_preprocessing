# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 18:45:17 2021

@author: yxzha

句子到句子，适用：翻译|问答等序列预测问题
N  : data sample-sentences-number
K1 : input_vocab_size 
K2 : output_vocab_size
T1 : max input-sequence-length
T2 : max output-sequence-length

"""


#import  tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,LSTM  # CuDNNLSTM 可以替换LSTM ，在cuda支持的GPU上训练时，会快几十倍

import numpy as npy

import data_prepare as dtpr


if __name__=='__main__':
    # 0 准备数据
    fit_datas, params, token_dicts= dtpr.data_prepare_processing(datas=None,N=10000)
    fit_datas.keys()
    encoder_input_data=fit_datas['encoder_input_data']
    decoder_input_data=fit_datas['decoder_input_data']
    decoder_output_data=fit_datas['decoder_output_data']
    N,T1,T2,K1,K2=params['N'],params['T1'],params['T2'],params['K1'],params['K2']
    dictx,dictx_revrs=token_dicts['x'],token_dicts['x_revrs'] # dictx_revrs 在fit 和 predict时都用不到额
    dicty,dicty_revrs=token_dicts['y'],token_dicts['y_revrs']
    
    ## 1 定义计算图 
    # 1) 编码器定义 encoder  define
    num_neurons=128
    encoder_inputs=Input(shape=(None,K1)) 
    encoder=LSTM(num_neurons,return_state=True) # 返回中间结果
    encoder_outputs,state_h,state_c=encoder(encoder_inputs) 
    encoder_states=(state_h,state_c)  # 思想向量，类似word-embeding-vector的作用
    
    # 2) 解码器定义 decoder define
    decoder_inputs=Input(shape=(None,K2))
    decoder_lstm=LSTM(num_neurons,return_sequences=True,return_state=True) #返回序列结果
    decoder_outputs,_,_=decoder_lstm(decoder_inputs,initial_state=encoder_states ) # 使用encoder的state结果
    decoder_dense=Dense(K2,activation='softmax')
    decoder_outputs=decoder_dense(decoder_outputs)
    
    # 3) 组合encoder-decoder model
    mdl=Model(inputs=[encoder_inputs,decoder_inputs],
              outputs=decoder_outputs,
              name='seq2seq_zhangyaxu') # 此时网络图已搭建好
    
    # 4) compile seq2seq model and fit
    '''
    #categorical_crossentropy:多分类问题使用交叉熵损失函数
    #rmsprop : root mean square prop :是adaGrad的改进，
               区别在于rmsprop在计算梯度时不是暴力累加平方梯度，而是分配了权重p来控制历史梯度的影响 r->p*r+(1-p)g^2
    '''
    mdl.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc']) 
    #
    batch_size=64
    epochs=100
    mdl.fit([encoder_input_data,decoder_input_data],
            decoder_output_data,
            batch_size=batch_size,
            epochs=epochs)
    
    ## 2  生成序列：  需要自己重构训练层的结构，然后进行重新组装用于predict序列
    #2.1 编码器模型
    encoder_mdl=Model(inputs=encoder_inputs,outputs=encoder_states) 
    #2.2 解码器模型
    thought_input=[Input(shape=(num_neurons,)),
                   Input(shape=(num_neurons,))]
    decoder_outputs,state_h,state_c=decoder_lstm(decoder_inputs,initial_state=thought_input)
    decoder_states=[state_h,state_c]
    decoder_outputs=decoder_dense(decoder_outputs)
    decoder_mdl=Model(inputs=[decoder_inputs]+thought_input,
                      outputs=[decoder_outputs]+decoder_states)
    
    ##2.3 预测  生成序列
    #  构建生成函数
    #encoder_mdl,decoder_mdl,T2,K2,reverse_dict_ytoken_index,start_token='\t',stop_token='\n'
    start_token='\t';stop_token='\n'
    def decode_seq(encoder_input_seq):
        target_seq=npy.zeros(shape=(1,1,K2))
        target_seq[0,0,dicty[start_token]]=1. # 不应该传入 stop_token
        generate_seq=[]
        thought=encoder_mdl.predict(encoder_input_seq)
        stop_condition=False
        while not stop_condition:
            output_ytoken,h,c=decoder_mdl.predict([target_seq]+thought)
            output_ytoken_idx=npy.argmax(output_ytoken[0,-1,:])
            generate_char=dicty_revrs[output_ytoken_idx]
            #print(generate_char)
            generate_seq=generate_seq+[generate_char]
            if (generate_char==stop_token or len(generate_seq)>T2):
                stop_condition=True
            target_seq=npy.zeros(shape=(1,1,K2))
            target_seq[0,0,output_ytoken_idx]=1.0
            thought=[h,c]
        return generate_seq
    
    # 3.2 传入 a sentence ，生成encoder-input data
    def sent_preprocessing(x_sentence_list):
        encoder_input_seq=npy.zeros((1,T1,K1),dtype='float32')
        for t1,dtx in enumerate(x_sentence_list):
            if dictx.get(dtx):
                encoder_input_seq[0,t1,dictx.get(dtx)]=1.0
        decoder_sentence=decode_seq(encoder_input_seq)
        return decoder_sentence
    
    acc_num=0;NT=1000
    error_sample=[]
    for i in range(NT):
        x_test,y_test=dtpr.analog_datas(N=1)[0]
        y_pred=sent_preprocessing(x_sentence_list=x_test)
        y_pred.remove('\n')
        if y_test==y_pred:
            acc_num+=1
        else:
            error_sample.append([y_test,y_pred])
            pass #print(y_test ,y_pred)
    print('预测序列的完全准确率 {0} % : '.format(acc_num/NT*100)) # 97.3% ： neurons=128，batch=64 ,epoch=100

        
