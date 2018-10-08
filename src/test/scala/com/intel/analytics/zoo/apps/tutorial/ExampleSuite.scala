package com.intel.analytics.zoo.apps.tutorial

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.apps.movie.DataPreprocess
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Dataset, SparkSession}
import org.scalatest.{BeforeAndAfter, FunSuite}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.rdd.RDD


class ExampleSuite extends FunSuite with BeforeAndAfter {

  var sc: SparkContext = _
  var spark: SparkSession = _
  var ratingsin: Dataset[String] = _

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("TrainWithNCF")
      .setMaster("local[8]")
    sc = new SparkContext(conf)
    spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init

  }


  test("ALS explicit") {
    val dataProcess = new DataPreprocess
    val (ratings, userCount, itemCount) = dataProcess.getDataDF()
    val ratingsin = spark.read.textFile("/Users/guoqiong/intelWork/projects/wrapup/recommendation/ml-1m")
    println(ratingsin.count())

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userIdIndex")
      .setItemCol("itemIdIndex")
      .setRatingCol("label")

    val model: ALSModel = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    // model.setColdStartStrategy("drop")
    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    // Generate top 10 movie recommendations for each user
    //   val userRecs = model.recommendForAllUsers(10)
    // Generate top 10 user recommendations for each movie
    // val movieRecs = model.recommendForAllItems(10)
  }


  test("implicit") {
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setImplicitPrefs(true)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
  }
}
