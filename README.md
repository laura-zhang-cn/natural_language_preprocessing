# natural_language_preprocessing
include many sub-algorithms for the field of NLP

0. 业内关于NLP比较流行和成熟的算法很多，gensim包里面包含的比较多    
word2vec doc2vec （词向量  文档向量）  
N-gram （语义检查）  
TF-IDF （主题提取，文本分类）   
LDA （主题提取）   
...  

1. 基于上下文计算词的关联性：words_relevance_based_on_context   
如切词会将 “补充”  “水分”  切分开，但他们的关联度高，所以可以关联到一起变成： “补充水分” 

2. 基于词性粘滞方向的有效词组识别：generate_effect_new_word  
由于词典的分词算法往往受限于词典的覆盖度，‘不满意’ 会被分成‘不’‘满意’ ，并可能存在大量没有包含在字典中而落单的词，    
为了正确理解语义，对于特定场景下，采用基于词性粘滞方向的有效词组识别算法。   

3. jaro-winkler-improve   
识别短文本的相似度，如品牌名称，这里实现了jaro-winkler算法，并提供了3种优化方案。    
