### 原理说明等，请参考Info files/下的PDF文档   

### 基本环境条件
算法使用spark-submit运行在spark集群环境中，  
集群环境必须可使用sparksql访问hadoop下的数据库，并具有某个hive数据库(database_name)的写入权限, database_name在params.py中配置  

### 其中：  
run_word_semantic_similar.sh 是总执行文件  
params.py  是参数配置文件  
其它py脚本文件均在sh执行文件中依次执行  
