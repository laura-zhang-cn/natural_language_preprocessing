#!/usr/bin/env bash
# run on 10.16.32.40
# crontab -e
# 30 17 */15 * * nohup sh /home/yaxuz/word_semantic_similar/run_word_semantic_similar.sh  > /home/yaxuz/logs/word_semantic_similar_log.log 2>&1 &
#

source /etc/profile

set -x #追踪一段代码的显示情况，执行后在整个脚本有效

dump_file=0

tpath=`dirname $0`
tdata_path='/home/jm/recommend/data'

function dump_file_data()
{
    # 1. dump newest file to local tmp file
    # 2. insert file version to the first line
    # 3. rename tmp file to target file
    hdfs_file='/hiveweb/recommend.db/homemain/word_semantic_similar'
    hadoop fs -getmerge ${hdfs_file}/* ${tdata_path}/word_semantic_similar.csv
    if [ $? -ne 0 ];then
        send_msg "Failed to dump word_semantic_similar from hdfs to local"
        exit 1
    fi
}


######################## Start here ##################
if [ -f "${tpath}/onrunning_word_semantic_similar" ]
then 
    echo "Last is still running, quit now"
    exit 1
fi

touch ${tpath}/onrunning_word_semantic_similar  #一是用于把已存在文件的时间标签更新为系统当前的时间（默认方式），它们的数据将原封不动地保留下来；二是用来创建新的空文件。

# 切词
/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
	--files /home/jm/recommend/script/jiebaNewWords.txt \
	--conf spark.yarn.queue=root.default \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/cut_word.py

# w2v相似度 & jaro距离
/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/word2vec_sim.py

# 初始化tfidf向量
/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/tfidf_vec_generate.py

# tfidf-vec 相似度
/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 80 \
    --executor-memory 2g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/tfidf_vec_sim.py

# 上下文关联系数  relevance
/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/relevance_on_context.py

# 语料丰富度：单调性 unrichness
/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/corpus_richness_alg.py

# 近义词识别：模型整合： w2v+tfidf+jaro+relevance +unrichness
/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/get_features.py

/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/final_sim_filter.py


# 近义词聚类: 密度收敛可达的聚类算法
/home/jm1/cdh/spark/bin/spark-submit \
    --driver-memory 8g \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 2 \
    --conf spark.storage.memoryFraction=0.3 \
    --conf spark.shuffle.memoryFraction=0.3 \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.kryoserializer.buffer.max=256 \
    --conf spark.kryoserializer.buffer=64 ${tpath}/sim_cluster.py

[ ${dump_file} -eq 1 ] && dump_file_data

echo " word_semantic_similar  finished!"
rm -f ${tpath}/onrunning_word_semantic_similar

exit 0


