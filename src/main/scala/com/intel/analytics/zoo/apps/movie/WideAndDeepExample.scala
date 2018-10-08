/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.apps.movie

import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import scopt.OptionParser
import org.apache.spark.rdd.RDD
import com.intel.analytics.zoo.examples.recommendation.Rating
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.types.DoubleType

case class User(userId: Int, gender: String, age: Int, occupation: Int)

case class Item(itemId: Int, title: String, genres: String)

case class WNDParams(val modelType: String = "wide_n_deep",
                     val inputDir: String = "/Users/guoqiong/intelWork/projects/wrapup/recommendation/ml-1m"
                    )

object WideAndDeepExample {
  def main(args: Array[String]): Unit = {
    val defaultParams = WNDParams()
    val parser = new OptionParser[WNDParams]("WideAndDeep Example") {
      opt[String]("modelType")
        .text(s"modelType")
        .action((x, c) => c.copy(modelType = x))
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
    }
    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: WNDParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("WideAndDeepExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratingsDF, userDF, itemDF, userCount, itemCount) =
      loadPublicData(sqlContext, params.inputDir)

    val bucketSize = 100
    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("occupation", "gender"),
      wideBaseDims = Array(21, 3),
      wideCrossCols = Array("age-gender"),
      wideCrossDims = Array(bucketSize),
      indicatorCols = Array("genres", "gender"),
      indicatorDims = Array(19, 3),
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(userCount, itemCount),
      embedOutDims = Array(64, 64),
      continuousCols = Array("age"))

    val wideAndDeep: WideAndDeep[Float] = WideAndDeep[Float](
      params.modelType,
      numClasses = 5,
      columnInfo = localColumnInfo,
      hiddenLayers = Array(20,10))

    val isImplicit = false
    val featureRdds =
      assemblyFeature(isImplicit, ratingsDF, userDF, itemDF, localColumnInfo, params.modelType)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      featureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = featureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    val optimizer = Optimizer(
      model = wideAndDeep,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 2800)

    val optimMethod = new Adam[Float](
      learningRate = 1e-3,
      learningRateDecay = 1e-6)

    optimizer
      .setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(20))
      .optimize()

    val results = wideAndDeep.predict(validationRdds)
    results.take(5).foreach(println)

    val resultsClass = wideAndDeep.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    val userItemPairPrediction: RDD[UserItemPrediction] = wideAndDeep.predictUserItemPair(featureRdds)
    userItemPairPrediction.take(50).foreach(println)

    val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)

    val pairPredictionsDF = sqlContext.createDataFrame(userItemPairPrediction).toDF()

    val out = pairPredictionsDF.join(ratingsDF, Array("userId", "itemId"))
      .select(col("prediction").cast(DoubleType), col("label").cast(DoubleType))

    val correctCounts = out.filter(col("prediction") === col("label")).count()
    out.show(10)
    val accuracy = correctCounts.toDouble / featureRdds.count()
    val mseUdf = udf((predition: Double, label: Double) => (predition - label) * (predition - label))

    val MSE: DataFrame = out.select("prediction", "label")
      .withColumn("MSE", mseUdf(col("prediction"), col("label"))).select("MSE")
      .agg(avg(col("MSE")))

    MSE.show()
    println(accuracy)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(out)
    println(s"Root-mean-square error = $rmse")

    println(s"mean-square error = " + rmse * rmse)
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String):
  (DataFrame, DataFrame, DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()
    val userDF = sqlContext.read.text(dataPath + "/users.dat").as[String]
      .map(x => {
        val line = x.split("::")
        User(line(0).toInt, line(1), line(2).toInt, line(3).toInt)
      }).toDF()
    val itemDF = sqlContext.read.text(dataPath + "/movies.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Item(line(0).toInt, line(1), line(2).split('|')(0))
      }).toDF()

    val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    (ratings, userDF, itemDF, userCount, itemCount)
  }

  // convert features to RDD[Sample[Float]]
  def assemblyFeature(isImplicit: Boolean = false,
                      ratingDF: DataFrame,
                      userDF: DataFrame,
                      itemDF: DataFrame,
                      columnInfo: ColumnFeatureInfo,
                      modelType: String): RDD[UserItemFeature[Float]] = {

    // age and gender as cross features, gender its self as wide base features
    val genderUDF = udf(Utils.categoricalFromVocabList(Array("F", "M")))
    val bucketUDF = udf(Utils.buckBucket(100))
    val genresList = Array("Crime", "Romance", "Thriller", "Adventure", "Drama", "Children's",
      "War", "Documentary", "Fantasy", "Mystery", "Musical", "Animation", "Film-Noir", "Horror",
      "Western", "Comedy", "Action", "Sci-Fi")
    val genresUDF = udf(Utils.categoricalFromVocabList(genresList))

    val userDfUse = userDF
      .withColumn("age-gender", bucketUDF(col("age"), col("gender")))
      .withColumn("gender", genderUDF(col("gender")))

    // genres as indicator
    val itemDfUse = itemDF
      .withColumn("genres", genresUDF(col("genres")))

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(ratingDF)
      negativeDF.unionAll(ratingDF.withColumn("label", lit(2)))
    }
    else ratingDF

    // userId, itemId as embedding features
    val joined = unioned
      .join(itemDfUse, Array("itemId"))
      .join(userDfUse, Array("userId"))
      .select(unioned("userId"), unioned("itemId"), col("label"), col("gender"), col("age"),
        col("occupation"), col("genres"), col("age-gender"))

    val rddOfSample = joined.rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, modelType))
    })
    rddOfSample
  }

}
