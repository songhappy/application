package com.intel.analytics.bigdl.apps.movie

import com.intel.analytics.bigdl.Module
import com.intel.analytics.zoo.apps.recommendation.{ModelParam, ModelUtils}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.{BeforeAndAfter, FunSuite}
import org.apache.spark.sql.functions._
import com.intel.analytics.zoo.apps.recommendation.Utils._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, MSECriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.zoo.apps.movie.DataPreprocess
import org.apache.spark.rdd.RDD

class TrainWithSummary extends FunSuite with BeforeAndAfter {
  var spark: SparkSession = _

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("Train")
      .setMaster("local[8]")
    spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init
  }


  test("mlp") {
    val movieDataProcess = new DataPreprocess

    val (indexedDF, userCount, itemCount) = movieDataProcess.getDataDF()
    val dataRdd = df2rddOfSample(indexedDF)

    dataRdd.cache()

    val Array(train_rdd, val_rdd) = dataRdd.randomSplit(Array(0.8, 0.2), seed = 1L)

    val modelParam = ModelParam(labels = 5)
    val recModel = new ModelUtils(modelParam)
    val model = recModel.ncf(userCount.toInt, itemCount.toInt)

    val optimizer = Optimizer(model = model,
      sampleRDD = train_rdd,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 800)

    val vMethods: Array[ValidationMethod[Float]] = Array(new MAE[Float](), new Loss[Float]())
    val trigger: Trigger = Trigger.everyEpoch
    val dataSet: RDD[Sample[Float]] = val_rdd

    optimizer.setValidation(trigger, dataSet, vMethods, 800)

    val trained_model = optimizer.optimize()

    println(trained_model.getName() +"," + trained_model.getClass)
    //    val optimizer = Optimizer(model = model, training, MSECriterion[Float](), batchSize = 100)
    //
    //    val vMethods: Array[ValidationMethod[Float]] = Array(new MAE[Float](), new Loss[Float])
    //    val trigger: Trigger = Trigger.everyEpoch
    //    val dataSet: RDD[Sample[Float]] = test
    //
    //    val trainSummary = new TrainSummary("/tmp/log", "carrer2_train")
    //    trainSummary.setSummaryTrigger("Loss", Trigger.maxEpoch(1))
    //
    //    optimizer.setTrainSummary(trainSummary)
    //
    //    optimizer.setValidation(trigger, dataSet, vMethods, 100)
    //
    //    val loss: Array[(Long, Float, Double)] = trainSummary.readScalar("Loss")
    //
    //    val trained_model = optimizer.optimize()

    val res: Array[(ValidationResult, ValidationMethod[Float])] = trained_model.evaluate(val_rdd, Array(new MAE[Float]().asInstanceOf[ValidationMethod[Float]]))
    res.foreach(r => println(s"${r._2} is ${r._1}"))
    println(res.mkString(","))
  }

  //    val (train_rdd, val_rdd, predictRdd, max_user_id, max_movie_id) = dataPreprocess.createTrainData(resume, job, application)
  //
  //    val model = recModel.build_model_ncf(max_user_id, max_movie_id, 10, Array(20, 10, 10, 10))
  //    train_rdd.count()
  //
  //    val optimizer = Optimizer(model = model, train_rdd, ClassNLLCriterion[Float](), batchSize = 100)
  //
  //    val vMethods: Array[ValidationMethod[Float]] = Array(new MAE[Float](), new Loss[Float])
  //    val trigger: Trigger = Trigger.everyEpoch
  //    val dataSet: RDD[Sample[Float]] = val_rdd
  //
  //    val trainSummary = new TrainSummary("/tmp/log", "carrer2_train")
  //    trainSummary.setSummaryTrigger("Loss", Trigger.maxEpoch(1))
  //
  //    optimizer.setTrainSummary(trainSummary)
  //
  //    optimizer.setValidation(trigger, dataSet, vMethods, 100)
  //
  //    val loss: Array[(Long, Float, Double)] = trainSummary.readScalar("Loss")
  //
  //    val trained_model = optimizer.optimize()
  //
  //    val res: Array[(ValidationResult, ValidationMethod[Float])] = trained_model.evaluate(val_rdd, Array(new MAE[Float]().asInstanceOf[ValidationMethod[Float]]))
  //    res.foreach(r => println(s"${r._2} is ${r._1}"))
  //    println(res.mkString(","))

  //    val res: RDD[Int] = trained_model.predictClass(predictRdd)
  //
  //    case class Job(resume_id: Int, job_id: Int, score: Float)
  //
  //    val finalResult = predictRdd.zip(res).map(x => {
  //
  //      val dataArray = x._1.getData()
  //      Job(dataArray(0).toInt, dataArray(1).toInt, x._2)
  //
  //    })
  //
  //    val userBaseRec = finalResult.groupBy(x => x.resume_id)
  //        .map(x => (x._1, x._2.toList.sortBy(x => x.score).reverse.map(x=> (x.job_id,x.score)).take(10)))
  //
  //    userBaseRec.take(10).foreach(println)

}
