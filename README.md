# natural_language_preprocessing
include many sub-algorithms for the field of NLP

0. 业内关于NLP比较流行和成熟的算法很多，gensim包里面包含的比较多  
word2vec doc2vec （词向量  文档向量）
N-gram （语义检查）
TF-IDF （主题提取）
LDA （主题提取）
...

1. 基于上下文计算词的关联性  
words_relevance_based_on_context

2. 由于词典的分词算法往往受限于词典的覆盖度，‘不满意’ 会被分成‘不’‘满意’ ，并可能存在大量没有包含在字典中而落单的词，  
为了正确理解语义，对于特定场景下，采用基于词性粘滞方向的有效词组识别算法。  
generate_effect_new_word_group.py



