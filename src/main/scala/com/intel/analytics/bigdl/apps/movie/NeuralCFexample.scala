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

package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import scopt.OptionParser

case class NeuralCFParams(val inputDir: String = "/Users/guoqiong/intelWork/projects/wrapup/recommendation/ml-1m",
                          val batchSize: Int = 2800,
                          val nEpochs: Int = 10,
                          val learningRate: Double = 1e-3,
                          val learningRateDecay: Double = 1e-6
                         )

case class Rating(userId: Int, itemId: Int, label: Int)

object NeuralCFexample {

  def main(args: Array[String]): Unit = {

    val defaultParams = NeuralCFParams()

    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: NeuralCFParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("NCFExample").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.getNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    // learningRate(0.001, 0.005, 0.01, 0.05)
    // learningRateDecay(1e-6, 1e-5, 1e-4, 1e-3)
    // batchSize (1024,  4096, 16384, 65536)
    // hidden layers (100, 50), (40, 20), (20, 10), (100, 40, 10)
    // nEpochs (5, 10, 20, 40)

    val learningRates = List(0.001, 0.005, 0.01, 0.05)
    val learningRateDecys = List(1e-6, 1e-5, 1e-4, 1e-3)
    val batchSizes = List(1024,  4096, 16384, 65536)
    val hiddenlayers = List(Array(100, 50), Array(40, 20), Array(20, 10), Array(100, 40, 10))
    val nEpochs = List(5,10, 20)

    val logs: Seq[String] = learningRates.flatMap(lr => {
      learningRateDecys.flatMap(lrd => {
        batchSizes.flatMap(bs => {
          hiddenlayers.flatMap(hdly => {
            nEpochs.map(epoch => runSingle(sqlContext, param, lr, lrd, bs, hdly, epoch))
          })
        })
      })
    })

    logs.foreach(println(_))
  }

  def runSingle(sqlContext: SQLContext, param: NeuralCFParams, learningRate: Double,
                learningRateDecay: Double, batchSize: Int, hidenlayers: Array[Int], nEpochs: Int) = {

    val (ratings, userCount, itemCount) = loadPublicData(sqlContext, param.inputDir)

    val isImplicit = false

    val pairFeatureRdds: RDD[UserItemFeature[Float]] =
      assemblyFeature(isImplicit, ratings, userCount, itemCount)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      pairFeatureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds: RDD[Sample[Float]] = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)
    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 5,
      hiddenLayers = hidenlayers)

    val optimizer = Optimizer(
      model = ncf,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = batchSize)

    val optimMethod = new Adam[Float](
      learningRate = learningRate,
      learningRateDecay = learningRateDecay)

    optimizer
      .setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(nEpochs))
      .optimize()

    val results = ncf.predict(validationRdds)
    results.take(5).foreach(println)
    val resultsClass = ncf.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    val pairPredictions = ncf.predictUserItemPair(validationpairFeatureRdds)
    val pairPredictionsDF = sqlContext.createDataFrame(pairPredictions).toDF()
    val out = pairPredictionsDF.join(ratings, Array("userId", "itemId"))
      .select(col("prediction").cast(DoubleType), col("label").cast(DoubleType))

    val correctCounts = out.filter(col("prediction") === col("label")).count()
    out.show(10)
    val accuracy = correctCounts.toDouble / validationpairFeatureRdds.count()
    val mseUdf = udf((predition: Double, label: Double) => (predition - label) * (predition - label))

    val MSE: DataFrame = out.select("prediction", "label")
      .withColumn("MSE", mseUdf(col("prediction"), col("label"))).select("MSE")
      .agg(avg(col("MSE")))

   // println("accuracy" + accuracy)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(out)
  //  println(s"Root-mean-square error = $rmse")

   // println(s"mean-square error = " + rmse * rmse)

    learningRate + "," + learningRateDecay + "," + batchSize + "," + hidenlayers.mkString("|") + "," + nEpochs + "," + accuracy + "," + rmse
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()

    val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    (ratings, userCount, itemCount)
  }

  def assemblyFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int): RDD[UserItemFeature[Float]] = {

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(indexed)
      negativeDF.unionAll(indexed.withColumn("label", lit(2)))
    }
    else indexed

    val rddOfSample: RDD[UserItemFeature[Float]] = unioned
      .select("userId", "itemId", "label")
      .rdd.map(row => {
      val uid = row.getAs[Int](0)
      val iid = row.getAs[Int](1)

      val label = row.getAs[Int](2)
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    rddOfSample
  }

}
