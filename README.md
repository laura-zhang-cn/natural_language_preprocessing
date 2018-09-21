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
  
4. 最长公共子序列改进：LCSS-unique  
对于有序队列的比较，一般采用提取最长公共子序列的算法，但原始的基于动态规划的算法 会导致调换两个待比较的字符串的位置后 获得的子序列不同，遂我们做了后续改进，会根据子序列在字符串中的连续程度，选择连续程度更高的子序列。  
