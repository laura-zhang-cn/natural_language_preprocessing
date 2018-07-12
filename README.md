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
![effect1](https://github.com/laura-zhang-cn/natural_language_preprocessing/blob/master/effect-images/concat_prop_word_effect.png)  
3. 识别短文本的相似度，如名称，这里实现了jaro-winkler算法，并提供了3种优化方案。  
流行方法 ： 1/3×(m/s1+m/s2+(m-t)/m)   #两个字符串中匹配度和匹配后的换位比例 相加，三者的平均值 ： 存在普遍偏高的问题  
改进1   :  1/2×(m/s1+m/s2) × (m-t)/m  #先求字符串匹配度的平均值，然后使用匹配后的换位比例做衰减，无换位则不衰减，最多衰减0.5  
改进3 ：   (w1×m/s1+w2×m/s2) × (m-t)/m #字符串匹配度的**加权**平均值 ，权重与字符串的长度相关，字符串越长，权重越偏高，且w1+w2=1  
jaro-winkler-improving.py  
mode='popular' 流行方法  
mode='improve1' 改进方法1  
mode='improve3' 改进方法3  
