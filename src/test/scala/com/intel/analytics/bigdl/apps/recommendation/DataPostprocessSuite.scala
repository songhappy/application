package com.intel.analytics.bigdl.apps.recommendation

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FunSuite}

case class User(id:Int, score:Double)

class DataPostprocessSuite extends FunSuite with BeforeAndAfter {

 var spark: SparkSession = _

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("Train")
      .setMaster("local[8]")
    spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init
  }

  test("metrics") {
    val sparkSession = SparkSession.builder().getOrCreate()
    import sparkSession.implicits._
    val data = Array((0, 0.1), (1, 0.8), (2, 0.2)).map( x=> User(x._1,x._2))
    val dataset = spark.createDataset(data)
    dataset.show(10)
    dataset.select("id").show()

    dataset.map(x=> x.score +1).show()
  }

  test("DF") {
   // sc.parallelize()
  }

}
