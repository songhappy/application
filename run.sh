~/intelWork/tools/spark/spark-2.2.0-bin-hadoop2.7/bin/spark-submit \
--master local[8] \
--driver-memory 16g \
--jars ./target/application-1.0-SNAPSHOT-job.jar \
--executor-memory 16g \
--class com.intel.analytics.zoo.examples.recommendation.NeuralCFexample \
./target/application-1.0-SNAPSHOT.jar
#--class com.intel.analytics.bigdl.apps.job2Career.DataAnalysis \
#--class com.intel.analytics.bigdl.apps.job2Career.DataProcess \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithALS \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithNCF \
