package com.intel.analytics.zoo.apps.movie

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions.{col, round}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

case class RatingAls(userId: Int, movieId: Int, rating: Float, timestamp: Long)

object TrainWithALS {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("example").set("spark.sql.crossJoin.enabled", "true")
    val sc = new SparkContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val spark = SparkSession.builder().getOrCreate()

    def parseRating(str: String): RatingAls = {
      val fields = str.split("::")
      assert(fields.size == 4)
      RatingAls(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
    }

    import spark.implicits._
    val ratings = spark.read.textFile("/Users/guoqiong/intelWork/projects/wrapup/recommendation/ml-1m/ratings.dat")
      .map(x => parseRating(x))
      .toDF()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(ratings)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    model.setColdStartStrategy("drop")
    val predictions = model.transform(ratings).withColumn("predict",round(col("prediction")))

    predictions.show()

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("predict")
    val rmse = evaluator.evaluate(predictions)

    val correctCounts = predictions.filter(col("predict") === col("rating")).count()
    predictions.show(10)
    val accuracy = correctCounts.toDouble / ratings.count()

    println(accuracy)
    println(s"Root-mean-square error = $rmse")

    println(s"mean-square error = " + rmse * rmse)

  }
}
