package com.intel.analytics.zoo.apps.anomalyDetection

import RNNModel.buildModel
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{avg, col}
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.joda.time.{DateTime, DateTimeField}
import org.joda.time.DateTime
import org.joda.time.format.DateTimeFormat
import org.joda.time.format.DateTimeFormatter
import scala.io.Source
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}

case class Rec(id: String, val1: String, val2: String, val3: String)

class RNNModelSpec extends ZooSpecHelper {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("AnomalyDetectionTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)

  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "simple test" should "work well " in {
    val input = Tensor[Float](Array(1, 20, 1)).rand()
    val model = buildModel(Shape(20, 1))
    val output = model.forward(input)
    val gradInput = model.backward(input, output)
  }

  "timestamp" should "work well" in {

    val formatter = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")
    val dt = formatter.parseDateTime("2018-05-23 00:00:00")
    val x: Int = dt.dayOfWeek().get()
    println(dt.dayOfWeek().get, dt.hourOfDay().get(), dt.monthOfYear().get(), dt.dayOfMonth().get())
  }

  "multipleArray" should "work well" in {
    val matrix = Array.ofDim[Int](2, 2)
    //unroll data into this matrix and convert into sample to train a model using rdd based api

  }

  "scaler" should "work properly" in {
    val inputPath = "/Users/guoqiong/intelWork/git/analytics-zoo/dist_2.x/bin/data/NAB/nyc_taxi"
    val df = NYCTaxiExample.loadPublicData(sqlContext, inputPath)
    df.show(10, false)
    df.describe().show()
    val mean: Array[Row] = df.select("value").agg(avg(col("value"))).collect()

    val scaled = Utils.standardScale(df, Seq("value", "hour", "awake"))
    scaled.show(10, false)


  }

  "distributeUnroll" should "scale into big data set" in {

    val data = sc.parallelize(1 to 10).map(x => Array(x.toFloat))

    val unrolldata = Utils.distributeUnrollAll(data, 3)

    unrolldata.take(10).foreach(println)

  }

  "load BaoSight data" should "work fine" in {

    // val input = LocalParams().inputDir
    val input = "/Users/guoqiong/intelWork/projects/BaoSight/IMS/2nd_test"

    val timestamp: RDD[String] = sc.textFile(input + "/timestamp.txt")
    timestamp.take(10).foreach(println)


    val dataIn = timestamp.map(x => {
      val lines: Array[Array[Float]] = Source.fromFile(input + "/" + x).getLines
        .map(x => x.split("\t").map(y => y.toFloat)).toArray

      val size = lines.length

      val means: Seq[Float] = (0 to lines(0).length - 1).map(y => lines.map(xx => xx(y)).sum / size)
      x + "," + means.mkString(",")
    })

    dataIn.take(10).foreach(println)

    dataIn.coalesce(1).saveAsTextFile("/Users/guoqiong/intelWork/projects/BaoSight/result/IMS/2nd_test/2nd")

    // val datain = sc.textFile(input +)

  }


  "sum of array" should "work fine" in {
    val arrays = Array(Array(1, 2, 3), Array(4, 5, 6))
    val means = (0 to arrays(0).length - 1).map(y => arrays.map(xx => xx(y)).sum / 2.0)
    println(means)
  }

  "sparkdataframe" should "work fine" in {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val data: RDD[String] = sc.parallelize(Seq("a,1,2,3", "b,1,2,3", "c,2,2,3"))
    data.take(10).foreach(println)
    val df = data.map(x => {
      val lines = x.split(",")
      Rec(lines(0), lines(1), lines(2), lines(3))
    }).toDF()
    df.show()
  }

}