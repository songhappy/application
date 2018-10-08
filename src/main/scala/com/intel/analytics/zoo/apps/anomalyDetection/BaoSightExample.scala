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

import scala.io.Source

case class LocalParams(val inputDir: String = "/Users/guoqiong/intelWork/projects/BaoSight/IMS/4th_test/txt",
                       val outputDir: String = "/Users/guoqiong/intelWork/projects/BaoSight/result/IMS/4th_test",
                       val batchSize: Int = 2800,
                       val nEpochs: Int = 30,
                       val learningRate: Double = 1e-3,
                       val learningRateDecay: Double = 1e-6)

case class Baosight(ts: String, value: Float)

object BaoSightExample {

  def main(args: Array[String]): Unit = {

    val defaultParams = LocalParams()

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

    val (train, test) = prepareData(df, 50, testdataSize = 500)

    val model = RNNModel.buildModel(Shape(50, 1))
    val trainRdd = toSampleRdd(sc, train)
    val testRdd = toSampleRdd(sc, test)

    // sc.stop()

    val batchSize = param.batchSize
    model.compile(loss = "mse", optimizer = "rmsprop")
    model.fit(trainRdd, batchSize = batchSize, nbEpoch = param.nEpochs)

    testRdd.take(10).map(x => {
      // println("Shape:" + x.getFeatureSize().map(x => x.mkString(",")).mkString("|"))
      // println("data:" + x.getData().mkString(","))
    })


    val predictions: RDD[Activity] = model.predict(trainRdd, batchSize = batchSize)

    println("size of test" + predictions.count())
    predictions.take(10).foreach(println)
    println("stop")

    val y_predict = predictions.map(x => x.toTensor.toArray()(0))

    train.zip(y_predict).map(x => x._1.label + "," + x._2.toFloat).saveAsTextFile(param.outputDir + "/test.csv")

    //valuePredicted, valueTest
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String) = {

    @transient lazy val formatter = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")

    val timestamp: RDD[String] = sqlContext.sparkContext.textFile(dataPath + "/timestamp.txt")
    timestamp.take(10).foreach(println)

    val dataIn = timestamp.map(x => {
      val lines = (Source.fromFile(dataPath + "/" + x).getLines
        .map(x => x.split("\t")(0).toFloat)).toArray
      val sum = lines.sum
      val size = lines.size
      val mean = sum / size
      Baosight(x, mean)
    })

    val df = sqlContext.createDataFrame(dataIn)
    df.show(10, false)

    println(df.rdd.partitions.length)

    val featureDF = df.drop("ts").select("value")

    featureDF.show(5, false)
    println("tatal count: " + featureDF.count())

    featureDF
  }

  def prepareData(df: DataFrame, unrollLength: Int, testdataSize: Int = 1000) = {

    val scaledDF = standardScale(df, Seq("value"))
    val dataRdd = scaledDF.rdd.map(row => Array
    (row.getAs[Double](0).toFloat)).coalesce(1)

    val unrollData: RDD[FeatureLabelIndex] = distributeUnrollAll(dataRdd, unrollLength)

    val cutPoint = unrollData.count() - testdataSize

    val x_train = unrollData.filter(x => x.index < cutPoint)
    val x_test = unrollData.filter(x => x.index >= cutPoint)

    (x_train, x_test)

  }
}