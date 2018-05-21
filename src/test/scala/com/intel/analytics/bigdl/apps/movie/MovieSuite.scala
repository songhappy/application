package com.intel.analytics.bigdl.apps.movie

import java.io.{FileWriter, PrintWriter}

import com.intel.analytics.bigdl.apps.recommendation
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.apps.recommendation._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.{DLClassifier, DLModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FunSuite}
import com.intel.analytics.bigdl.apps.recommendation.ModelUtils
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType

case class User(id: Int, movieId: Int, score: Float)

case class dfUser(id: Int, movieId: Int, score: Float)

class MovieSuite extends FunSuite with BeforeAndAfter {

  var sc: SparkContext = _
  var spark: SparkSession = _

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("Train")
      .setMaster("local[8]")
    sc = new SparkContext(conf)
    spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init
  }

  test("explicitRdd") {

    val movieDataProcess = new DataPreprocess

    val (indexedDF, userCount, itemCount) = movieDataProcess.getDataDF()
    indexedDF.show(5)
    indexedDF.groupBy("label").count().show()
    println(userCount, itemCount)

    val indexedRdd = df2rddOfSample(indexedDF)
    indexedRdd.cache()

    val Array(train, test) = indexedRdd.randomSplit(Array(0.8, 0.2))

    train.count()

    val modelParam = ModelParam(labels = 5)
    val model = new ModelUtils(modelParam).mlpSeq(userCount.toInt, itemCount.toInt)
    val criterion = ClassNLLCriterion()
    val optimizer = Optimizer(model = model,
      sampleRDD = train,
      criterion = criterion,
      batchSize = 2800)

    val vMethods: Array[ValidationMethod[Float]] = Array(new MAE[Float](), new Loss[Float]())
    val trigger: Trigger = Trigger.everyEpoch

    optimizer.setValidation(trigger, test, vMethods, 2800)

    val trainedModel = optimizer.optimize()

    val predictedLabels: RDD[Int] = trainedModel.predictClass(test)

    predictedLabels.take(10).foreach(println)

    //    val predictions = test.zip(predictedLabels).map(x => {
    //      val dataArray: Array[Float] = x._1.getData()
    //      movie.User(dataArray(0).toInt, dataArray(1).toInt, x._2)
    //      // (dataArray(0).toInt, dataArray(1).toInt, x._2)
    //    })
    //
    //    val recommendations: RDD[(Int, List[User])] = predictions.groupBy(x => x.id)
    //      .map(x => (x._1, x._2.toList.sortBy(x => x.score).reverse.take(10)))
    //
    //    recommendations.take(10).foreach(println)
  }

  //
  //  test("implicit DF") {
  //
  //    val movieDataProcess = new DataPreprocess
  //
  //    val (indexedDF, userCount, itemCount) = movieDataProcess.getDataDF()
  //
  //    val dataWithNegative = addNegativeSample(1, indexedDF.withColumn("label", lit(1.0d)))
  //      .withColumn("label", add1(col("label")))
  //
  //    val dataInLP: DataFrame = df2LP(dataWithNegative)
  //
  //    val Array(trainingDF, validationDF) = dataInLP.randomSplit(Array(0.8, 0.2), seed = 1L)
  //
  //    trainingDF.cache()
  //    trainingDF.show(3)
  //
  //    val time1 = System.nanoTime()
  //
  //    //    val userEmbeds = Array(200, 100, 40, 10)
  //    //    val itemEmbeds = Array(200, 100, 40, 10)
  //    //    val laybers = Array(Array(200, 100, 40, 10), Array(100, 40, 10), Array(40, 10))
  //
  //    val userEmbeds = Array(10)
  //    val itemEmbeds = Array(10)
  //    val laybers = Array(Array(10))
  //    val fw = new FileWriter("/Users/guoqiong/intelWork/projects/jobs2Career/results/evaluation.txt", true)
  //
  //    userEmbeds.map(userEmbed => {
  //      itemEmbeds.map(itemEmbed => {
  //        laybers.map(layer => {
  //
  //          val modelParam = ModelParam(userEmbed = userEmbed,
  //            itemEmbed = itemEmbed,
  //            midLayers = layer,
  //            labels = 2)
  //
  //          val recModel = new ModelUtils(modelParam)
  //
  //          val model = recModel.mlp(userCount.toInt, itemCount.toInt)
  //
  //          val criterion = ClassNLLCriterion()
  //          //val criterion = MSECriterion[Float]()
  //
  //          val dlc = new DLClassifier(model, criterion, Array(2))
  //            .setBatchSize(1000)
  //            .setOptimMethod(new Adam())
  //            .setLearningRate(1e-3)
  //            .setLearningRateDecay(1e-7)
  //            .setMaxEpoch(3)
  //
  //          val dlModel: DLModel[Float] = dlc.fit(trainingDF)
  //
  //          trainingDF.unpersist()
  //
  //          println("featuresize " + dlModel.featureSize)
  //          println("model weights  " + dlModel.model.getParameters())
  //          val time2 = System.nanoTime()
  //
  //          val predictions: DataFrame = dlModel.setBatchSize(1).transform(validationDF)
  //
  //          val time3 = System.nanoTime()
  //
  //          predictions.cache()
  //          predictions.printSchema()
  //          predictions.show(3)
  //
  //          val res = Evaluation.evaluate(predictions.withColumn("label", toZero(col("label")))
  //            .withColumn("prediction", toZero(col("prediction"))))
  //
  //          val resStr = modelParam + "\n" + res.mkString(" | ") + "\n---------------"
  //
  //          println(resStr)
  //
  //          fw.write(resStr + "\n")
  //          fw.flush()
  //
  //          validationDF.unpersist()
  //
  //          val time4 = System.nanoTime()
  //
  //          val trainingTime = (time2 - time1) * (1e-9)
  //          val predictionTime = (time3 - time2) * (1e-9)
  //          val evaluationTime = (time4 - time3) * (1e-9)
  //
  //          println("training time(s):  " + toDecimal(3)(trainingTime))
  //          println("prediction time(s):  " + toDecimal(3)(predictionTime))
  //          println("evaluation time(s):  " + toDecimal(3)(predictionTime))
  //
  //          val usersForRec = indexedDF.select("userIdIndex").limit(10)
  //          val itemsForRec = indexedDF.select("itemIdIndex").limit(10)
  //
  //          usersForRec.show()
  //          usersForRec.join(indexedDF).show(10)
  //          println(usersForRec.join(indexedDF).count())
  //
  //          val userRec = recModel.recommendForUsers(usersForRec, indexedDF.select("itemIdIndex").distinct(), indexedDF, dlModel, 3)
  //
  //          userRec.printSchema()
  //          userRec.show(false)
  //
  //          val itemRec = recModel.recommendForUsers(indexedDF.select("userIdIndex").distinct(), itemsForRec, indexedDF, dlModel, 3)
  //          itemRec.printSchema()
  //          itemRec.show(false)
  //
  //        })
  //      })
  //    })
  //
  //    fw.close()
  //  }
  //
  ////
  test("implicitRdd") {

    val movieDataProcess = new DataPreprocess

    val (indexedDF, userCount, itemCount) = movieDataProcess.getDataDF()
    import indexedDF.sparkSession.implicits._

    val indexedWithNegative = addNegativeSample1(indexedDF.withColumn("label", lit(1.0d)))
      .withColumn("label", add1(col("label")))

    indexedWithNegative.show(5)
    indexedWithNegative.groupBy("label").count.show()

    val indexedRdd = df2rddOfSample(indexedWithNegative)

    indexedRdd.cache()
    val Array(train, test) = indexedRdd.randomSplit(Array(0.8, 0.2))
    train.count()

    val modelParam = ModelParam(labels = 2)
    val model = new ModelUtils(modelParam).mlpSeq(userCount.toInt, itemCount.toInt)

    val criterion = ClassNLLCriterion()

    val optimizer = Optimizer(model = model,
      sampleRDD = train,
      criterion = criterion,
      batchSize = 2800)

    val vMethods: Array[ValidationMethod[Float]] = Array(new MAE[Float](), new Loss[Float]())
    val trigger: Trigger = Trigger.everyEpoch

    optimizer.setValidation(trigger, test, vMethods, 2800)

    val trainedModel = optimizer.optimize()

    val scores: RDD[Activity] = trainedModel.predict(test)

    scores.take(10).foreach(println)
    //
    //    val predictions: DataFrame = test.zip(scores)
    //      .map(x => {
    //        val dataArray: Array[Float] = x._1.getData()
    //        (dataArray(0).toDouble, dataArray(1).toDouble, dataArray(2).toDouble, x._2.toTensor.value().toDouble)
    //      }).toDF("userIdIndex", "itemIdIndex", "label", "score")
    //
    //    predictions.show()
    //
    //    predictions.groupBy("label").count().show()
    //
    //    val precisionRecall = (0.1 to 0.9 by 0.05)
    //      .map(x => calculatePreRec(x.toFloat, predictions))
    //      .sortBy(-_._1)
    //
    //    precisionRecall.foreach(x => println(x.toString))

    //    val decile = score2bucket(predictions)
    //      .orderBy(col("bucket").desc)
    //
    //    decile.show

    // Evaluation.evaluate(predictions)
  }

  test("dataframe") {

//    val movieDataProcess = new DataPreprocess
//
//    val (indexedDF, userCount, itemCount) = movieDataProcess.getDataDF()
//    indexedDF.show(5)
//    println(indexedDF.count())
//
//    val schema: StructType = indexedDF.schema
//    val empty = indexedDF.sqlContext.createDataFrame(sc.emptyRDD, schema)
//
//    val full = indexedDF.union(empty)
//    println(full.count())
//    full.show(5)
//


  }

}
