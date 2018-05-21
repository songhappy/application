package com.intel.analytics.bigdl.apps.mc

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FunSuite}
import com.intel.analytics.bigdl.apps.mc.DataPreprocess._
import org.apache.spark.sql.functions._

class UtilsSuite extends FunSuite with BeforeAndAfter {


  var spark: SparkSession = _
  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("TrainWithNCF")
      .setMaster("local[8]")
    spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init
  }

  val inputpath = "/Users/guoqiong/intelWork/projects/mc/data"

  test("data analysis") {
    val raw = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .csv(inputpath + "/res_purchase_card_(pcard)_fiscal_year_2014_3pcd-aiuu.csv").cache()

    raw.show()
    raw.printSchema()
    println("total records: " + raw.count())
    raw.select("Agency Name").distinct().show(false)
    raw.groupBy("Cardholder Last Name", "Cardholder First Initial").count().show()

    //    import spark.implicits._
    //    raw.filter($"Cardholder Last Name" === "Rowland").show()
    //    raw.filter(col("Cardholder Last Name").equals("Rowland")).filter("Cardholder First Initial='C'").show(200)

    println("distinct vendors: " + raw.select("Vendor").distinct().count())

    val toDate = udf { s: String =>
      val splits = s.substring(0, 10).split("/")
      val year = splits(2)
      val month = splits(0)
      val day = splits(1)
      year + month + day
    }
    val dates = raw.select("Transaction Date").withColumn("date", toDate(col("Transaction Date")))
    dates.select(min("date")).show(false)
    dates.select(max("date")).show(false)

  }


}