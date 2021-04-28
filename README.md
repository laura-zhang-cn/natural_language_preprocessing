# natural_language_preprocessing
include many sub-algorithms for the field of NLP

0. 业内关于NLP比较流行和成熟的算法很多，gensim包里面包含的比较多    
word2vec doc2vec （词向量  文档向量）  
N-gram （语义检查）  
TF-IDF （主题提取，文本分类）   
LDA （主题提取）   
...  
以上请去gensim的官方文档查看使用方法。  
> ## 本分支主要包括：自定义或改进的nlp算法  
1. 基于上下文计算词的关联性：words_relevance_based_on_context   
如切词会将 “补充”  “水分”  切分开，但他们的关联度高，所以可以关联到一起变成： “补充水分” 

2. 基于词性粘滞方向的有效词组识别：generate_effect_new_word  
由于词典的分词算法往往受限于词典的覆盖度，‘不满意’ 会被分成‘不’‘满意’ ，并可能存在大量没有包含在字典中而落单的词，    
为了正确理解语义，对于特定场景下，采用基于词性粘滞方向的有效词组识别算法。   

3. jaro-winkler-improve   
识别短文本的相似度，如品牌名称，这里实现了jaro-winkler算法，并提供了3种优化方案。    
  
4. 最长公共子序列改进：LCSS-optimization  
对于有序队列的比较，一般采用提取最长公共子序列的算法，但原始的基于动态规划的算法 会导致调换两个待比较的字符串的位置后 获得的子序列不同，遂我们做了后续改进，会根据子序列在字符串中的连续程度，选择连续程度更高的子序列。  

5. 近义词识别算法 ：synonyms_fuse_model     
5.1)  融合(Word2Vec、TF-IDF、jaro、context-relevance、corpus-unrichness) ，计算词的语义相似性  
整体准确率在 **88%** 左右，对比了目前流行的synonyms，效果要好很多    
![近义词识别效果](https://github.com/laura-zhang-cn/natural_language_preprocessing/blob/master/effect-images/%E8%BF%91%E4%B9%89%E8%AF%8D%E8%AF%86%E5%88%AB%E6%95%88%E6%9E%9C.png)  
5.2)  对于语义相似的词，使用密度收敛可达的聚类算法(anti-DBSCAN)   
![密度可达的聚类算法效果](https://github.com/laura-zhang-cn/natural_language_preprocessing/blob/master/effect-images/%E5%AF%86%E5%BA%A6%E6%94%B6%E6%95%9B%E5%8F%AF%E8%BE%BE%E7%9A%84%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95.png)  
5.3)  仍可以继续优化的地方：  
12%的误差主要来自于以下几类词：   
> 季节、时间 ： 春夏秋冬、早中晚、日夜；  
> 大小、高矮、数字性的： 七分、九分、中筒、长筒；  
> 颜色： 黑白红灰；   
> 形状： 方框、圆框、方跟、圆跟、粗跟、细跟;   
> 方向、方位： 前中后、首尾、上下左右、顶 底；   
> 部位： 牙齿、牙龈、头发、头皮、插头、插座、镜片、镜框、鞋头、鞋身、杯身、杯底；  


