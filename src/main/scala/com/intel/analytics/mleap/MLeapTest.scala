//package com.intel.analytics.mleap
//
//import ml.combust.bundle.BundleFile
//import ml.combust.bundle.dsl.Bundle
//import ml.combust.mleap.spark.SparkSupport._
//import ml.combust.mleap.runtime.MleapSupport._
//import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row, Transformer}
//import ml.combust.mleap.core.types._
//import org.apache.spark.SparkConf
//import org.apache.spark.ml.{Pipeline, PipelineModel}
//import org.apache.spark.ml.bundle.SparkBundleContext
//import org.apache.spark.ml.feature.{Binarizer, StringIndexer}
//import org.apache.spark.sql._
//import org.apache.spark.sql.functions._
//import resource._
//import org.apache.log4j.Logger
//import org.apache.log4j.Level
//
//object MLeapTest {
//
//  def main(args: Array[String]): Unit = {
//
//    Logger.getLogger("org").setLevel(Level.OFF)
//    Logger.getLogger("akka").setLevel(Level.OFF)
//
//    val spark = SparkSession.builder().master("local[8]").getOrCreate()
//
//    val datasetName = "src/test/data/mleapexample/spark-demo.csv"
//
//    val dataframe: DataFrame = spark.sqlContext.read.format("csv")
//      .option("header", true)
//      .load(datasetName)
//      .withColumn("test_double", col("test_double").cast("double"))
//    dataframe.show()
//
//    // User out-of-the-box Spark transformers like you normally would
//    val stringIndexer = new StringIndexer().
//      setInputCol("test_string").
//      setOutputCol("test_index")
//
//    val binarizer: Binarizer = new Binarizer().
//      setThreshold(0.5).
//      setInputCol("test_double").
//      setOutputCol("test_bin")
//
//    val pipelineEstimator: Pipeline = new Pipeline()
//      .setStages(Array(stringIndexer, binarizer))
//
//    val pipeline: PipelineModel = pipelineEstimator.fit(dataframe)
//    val outputDF = pipeline.transform(dataframe)
//    outputDF.show()
//
//    // then serialize pipeline
//    val sbc = SparkBundleContext().withDataset(pipeline.transform(dataframe))
//
//
//    val bfiles = managed(BundleFile("jar:file:/Users/guoqiong/intelWork/git/application/src/test/model/simple-spark-pipeline.zip"))
//    for (bf <- managed(BundleFile("jar:file:/Users/guoqiong/intelWork/git/application/src/test/model/simple-spark-pipeline.zip"))) {
//      pipeline.writeBundle.save(bf)(sbc).get
//    }
//
//    val bundle: Bundle[Transformer] = (for (bundleFile <-bfiles) yield {
//      bundleFile.loadMleapBundle().get
//    }).opt.get
//
//    // MLeap makes extensive use of monadic types like Try
//    val schema: StructType = StructType(StructField("test_string", ScalarType.String),
//      StructField("test_double", ScalarType.Double)).get
//    val data = Seq(Row("hello", 0.6), Row("MLeap", 0.2))
//    val frame = DefaultLeapFrame(schema, data)
//
//    // transform the dataframe using our pipeline
//    val mleapPipeline = bundle.root
//    val frame2: DefaultLeapFrame = mleapPipeline.transform(frame).get
//    val data2: Seq[Row] = frame2.dataset
//
//    data2.map(row => println(row))
////
////    // get data from the transformed rows and make some assertions
////    assert(data2(0).getDouble(2) == 1.0) // string indexer output
////    assert(data2(0).getDouble(3) == 1.0) // binarizer output
////
////    // the second row
////    assert(data2(1).getDouble(2) == 2.0)
////    assert(data2(1).getDouble(3) == 0.0)
//
//  }
//}
