#-*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:00:29 2018

@author: zhangyaxu

配置参数表
"""

params = {
'shared': {
    'database_name': 'recommend',
    'doc_cut_word_table':'product_name_cut_words',
    'topic_cut_word_table':'cat_product_name_cut_words',
    'windowx': 3,
    'sizex': 100},
'word2vec': {
    'topn': 10,
    'sim_min': 0.5},
'tfidf': {
    'na_val': 0.0,
    'norm_type': 'maxscale',
    'topn': 50,
    'sim_min': 0.55},
'jaro': {
    'modex': 'improve3'},
'final_sim': {
    'kfold_use': False,
    'version': 'v2'}}

if __name__=='__main__':
    pass
