package com.intel.analytics.zoo.apps.anomalyDetection

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.common.NNContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.format.DateTimeFormat
import scopt.OptionParser
import com.intel.analytics.zoo.apps.anomalyDetection.Utils._

case class Taxi(ts: String, value: Float)

object NYCTaxiExample {

  def main(args: Array[String]): Unit = {

    val defaultParams = LocalParams(inputDir = "/Users/guoqiong/intelWork/git/analytics-zoo/dist_2.x/bin/data/NAB/nyc_taxi")

    val parser = new OptionParser[LocalParams]("AnomalyDetection Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("outputDir")
        .text(s"outputDir")
        .action((x, c) => c.copy(outputDir = x))
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

  def run(param: LocalParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("AnomalyDetection").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)
    val df = loadPublicData(sqlContext, param.inputDir)

    val (train, test) = prepareData(df, 50)


    val model = RNNModel.buildModel(Shape(50, 3))
    val trainRdd = toSampleRdd(sc, train)
    val testRdd = toSampleRdd(sc, test)

    // sc.stop()

    val batchSize = 2048
    model.compile(loss = "mse", optimizer = "rmsprop")
    model.fit(trainRdd, batchSize = batchSize, nbEpoch = 30)

    testRdd.take(10).map(x => {
      println("Shape:" + x.getFeatureSize().map(x => x.mkString(",")).mkString("|"))
      println("data:" + x.getData().mkString(","))
    })

    val predictions: RDD[Activity] = model.predict(testRdd, batchSize = batchSize)

    println("size of test" + predictions.count())
    predictions.take(10).foreach(println)
    println("stop")

    val y_predict = predictions.map(x => x.toTensor.toArray()(0))

    //val remove = "rm -rf " + param.outputDir + "/nyc/test.csv".!
    //println(remove)

    test.zip(y_predict).map(x => x._1.label + "," + x._2.toFloat).saveAsTextFile(param.outputDir + "/nyc/test.csv")

    //valuePredicted, valueTest
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String) = {

    @transient lazy val formatter = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")

    import sqlContext.implicits._
    val df = sqlContext.read.text(dataPath + "/nyc_taxi.csv").as[String]
      .rdd.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter
    }
      .map(x => {
        val line = x.split(",")
        Taxi(line(0), line(1).toFloat)
      }).toDF()
    // hour, awake (hour>=6 <=23)
    val hourUdf = udf((time: String) => {
      val dt = formatter.parseDateTime(time)
      dt.hourOfDay().get()
    })

    val awakeUdf = udf((hour: Int) => if (hour >= 6 && hour <= 23) 1 else 0)

    val featureDF = df.withColumn("hour", hourUdf(col("ts")))
      .withColumn("awake", awakeUdf(col("hour")))
      .drop("ts")
      .select("value", "hour", "awake")

    featureDF.show(5, false)
    println("tatal count: " + featureDF.count())

    featureDF
  }

  def prepareData(df: DataFrame, size: Int) = {

    val scaledDF = standardScale(df, Seq("value", "hour", "awake"))
    val dataRdd = scaledDF.rdd.map(row => Array
    (row.getAs[Double](0).toFloat, row.getAs[Double](1).toFloat, row.getAs[Double](2).toFloat))

    val testdataSize = 1000
    val unroll_length = 50

    val unrollData: RDD[FeatureLabelIndex] = distributeUnrollAll(dataRdd, unroll_length)

    val cutPoint = unrollData.count() - testdataSize

    val x_train = unrollData.filter(x => x.index < cutPoint)
    val x_test = unrollData.filter(x => x.index >= cutPoint)

    (x_train, x_test)
  }

}