export HADOOP_CONF_DIR=/etc/hadoop/conf
export SPARK_HOME=/opt/work/spark-2.4.3-bin-hadoop2.7
export ENV_HOME=/opt/work/guoqiong 
$SPARK_HOME/bin/spark-submit \
--master yarn \
--driver-memory 10g \
--executor-memory 64g \
--executor-cores 8 \
--num-executors 4 \
--class com.intel.analytics.friesian.Preprocess \
./target/application.jar  \
--input hdfs://172.16.0.105:8020/amz_review/ \
--outputDir hdfs://172.16.0.105:8020/user/root/guoqiong/friesian/scala/

#--class com.intel.analytics.bigdl.apps.job2Career.DataAnalysis \
#--class com.intel.analytics.bigdl.apps.job2Career.DataProcess \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithALS \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithNCF \
