//package com.intel.analytics.bigdl.apps.recommendation
//
//import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove
//import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.loadWordVecMap
//import com.intel.analytics.bigdl.apps.recommendation.Utils._
//import com.intel.analytics.bigdl.apps.recommendation.{Evaluation, ModelParam, ModelUtils}
//import com.intel.analytics.bigdl.nn._
//import com.intel.analytics.bigdl.optim.Adam
//import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
//import com.intel.analytics.bigdl.utils.Engine
//import org.apache.hadoop.conf.Configuration
//import org.apache.hadoop.fs.Path
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.SparkContext
//import org.apache.spark.broadcast.Broadcast
//import org.apache.spark.ml.{DLClassifier, DLModel}
//import org.apache.spark.sql.{SaveMode, SparkSession}
//import org.apache.spark.sql.functions._
//import org.scalatest.{BeforeAndAfter, FunSuite}
//
//class IncrementalTrainingSuite extends FunSuite with BeforeAndAfter{
//  var sc: SparkContext = _
//  var spark: SparkSession = _
//
//  before {
//    Logger.getLogger("org").setLevel(Level.ERROR)
//    val conf = Engine.createSparkConf().setAppName("Train")
//      .setMaster("local[8]")
//    sc = new SparkContext(conf)
//    spark = SparkSession.builder().config(conf).getOrCreate()
//    spark.sparkContext.setLogLevel("ERROR")
//    Engine.init
//  }
//
//  test("mlp incremental") {
//    //problematic since the prediction is not correct
//    //   val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed"
//    val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10"
//    val indexed = spark.read.parquet(input + "/indexed")
//    val userCount = spark.read.parquet(input + "/userDict").select("userIdIndex").distinct().count().toInt
//    val itemCount = spark.read.parquet(input + "/itemDict").select("itemIdIndex").distinct().count().toInt
//
//    val dataWithNegative = addNegativeSample(1, indexed)
//      .withColumn("userIdIndex", add1(col("userIdIndex")))
//      .withColumn("itemIdIndex", add1(col("itemIdIndex")))
//      .withColumn("label", add1(col("label")))
//
//    val dataInLP = df2LP(dataWithNegative)
//
//    val Array(trainingDF, validationDF) = dataInLP.randomSplit(Array(0.8, 0.2), seed = 1L)
//
//    trainingDF.cache()
//    validationDF.cache()
//
//    trainingDF.show(3)
//
//    val time1 = System.nanoTime()
//    val modelParam = ModelParam(userEmbed = 20,
//      itemEmbed = 20,
//      path="/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10/modelSeq",
//      midLayers = Array(40, 20),
//      labels = 2)
//
//    val recModel = new ModelUtils(modelParam)
//
//    // val model = recModel.ncf(userCount, itemCount)
//    val p = new Path(modelParam.path)
//    val fs = p.getFileSystem(new Configuration())
//
//    val model = if(!fs.exists(p)) {
//      recModel.ncf(userCount, itemCount)
//    }
//    else {
//      val m = Module.loadModule(modelParam.path)
//      println("-------------load model -----------------")
//      println(m)
//      println(m.getParameters()._1)
//      println(m.getParameters()._2)
//      println("----------------end---------------------")
//
//
//      m
//    }
//
//
//
//    val criterion = ClassNLLCriterion()
//    //val criterion = MSECriterion[Float]()
//
//    val dlc = new DLClassifier(model, criterion, Array(2))
//      .setBatchSize(1000)
//      .setOptimMethod(new Adam())
//      .setLearningRate(1e-2)
//      .setLearningRateDecay(1e-5)
//      .setMaxEpoch(10)
//
//
//    val dlModel: DLModel[Float] = dlc.fit(trainingDF)
//
//    println("featuresize " + dlModel.featureSize)
//    println("model weights  " + dlModel.model.getParameters())
//    val time2 = System.nanoTime()
//
//    model.saveModule(modelParam.path,true)
//
//    val predictions = dlModel.setBatchSize(1).transform(validationDF)
//
//    val time3 = System.nanoTime()
//
//    predictions.cache()
//    predictions.show(3)
//    predictions.printSchema()
//
//
//    Evaluation.evaluate(predictions.withColumn("label", toZero(col("label")))
//      .withColumn("prediction", toZero(col("prediction"))))
//
//    val time4 = System.nanoTime()
//
//    val trainingTime = (time2 - time1) * (1e-9)
//    val predictionTime = (time3 - time2) * (1e-9)
//    val evaluationTime = (time4 - time3) * (1e-9)
//
//    println("training time(s):  " + toDecimal(3)(trainingTime))
//    println("prediction time(s):  " + toDecimal(3)(predictionTime))
//    println("evaluation time(s):  " + toDecimal(3)(predictionTime))
//  }
//}
