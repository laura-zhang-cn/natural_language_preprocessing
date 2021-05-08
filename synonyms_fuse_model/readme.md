### 原理说明等，请参考Info files/下的PDF文档   

### 基本环境条件
算法使用spark-submit运行在spark集群环境中，  
集群环境必须可使用sparksql访问hadoop下的数据库，并具有某个hive数据库(database_name)的写入权限, database_name在params.py中配置  

### 其中：  
run_word_semantic_similar.sh 是总执行文件  
params.py  是参数配置文件  
其它py脚本文件均在sh执行文件中依次执行  

### 效果
1)  融合(Word2Vec、TF-IDF、jaro、context-relevance、corpus-unrichness) ，计算词的语义相似性  
整体准确率在 **88%** 左右，对比了目前流行的synonyms，效果要好很多    
![近义词识别效果](https://github.com/laura-zhang-cn/natural_language_preprocessing/blob/master/effect-images/%E8%BF%91%E4%B9%89%E8%AF%8D%E8%AF%86%E5%88%AB%E6%95%88%E6%9E%9C.png)  
2)  对于语义相似的词，使用密度收敛可达的聚类算法(anti-DBSCAN)   
![密度可达的聚类算法效果](https://github.com/laura-zhang-cn/natural_language_preprocessing/blob/master/effect-images/%E5%AF%86%E5%BA%A6%E6%94%B6%E6%95%9B%E5%8F%AF%E8%BE%BE%E7%9A%84%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95.png)  
3)  仍可以继续优化的地方：  
12%的误差主要来自于以下几类词：   
> 季节、时间 ： 春夏秋冬、早中晚、日夜；  
> 大小、高矮、数字性的： 七分、九分、中筒、长筒；  
> 颜色： 黑白红灰；   
> 形状： 方框、圆框、方跟、圆跟、粗跟、细跟;   
> 方向、方位： 前中后、首尾、上下左右、顶 底；   
> 部位： 牙齿、牙龈、头发、头皮、插头、插座、镜片、镜框、鞋头、鞋身、杯身、杯底；  

