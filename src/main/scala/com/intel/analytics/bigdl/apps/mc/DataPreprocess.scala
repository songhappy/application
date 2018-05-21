package com.intel.analytics.bigdl.apps.mc

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DateType, DoubleType}
import org.apache.spark.ml.feature.StringIndexer

object DataPreprocess {

  def prepareData(): DataFrame = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().master("local[1]").appName("test").getOrCreate()

    val raw = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .csv("data/res_purchase_card_(pcard)_fiscal_year_2014_3pcd-aiuu.csv").cache()

    val dataDF = raw.select("Cardholder Last Name", "Cardholder First Initial", "Amount", "Vendor", "Transaction Date", "Merchant Category Code (MCC)")
      .select(
        concat(col("Cardholder Last Name"), lit(" "), col("Cardholder First Initial")).as("name"),
        col("amount").cast(DoubleType),
        col("Vendor"),
        col("Transaction Date"),
        col("Merchant Category Code (MCC)").as("mccStr")
      )

    val toDate = udf { s: String =>
      val splits = s.substring(0, 10).split("/")
      val year = splits(2).toInt
      val month = splits(0).toInt
      val day = splits(1).toInt
      year * 10000 + month * 100 + day
    }
    val dates = dataDF.withColumn("date", toDate(col("Transaction Date"))).drop("Transaction Date")

    val si1 = new StringIndexer().setInputCol("name").setOutputCol("uid")
    val si2 = new StringIndexer().setInputCol("Vendor").setOutputCol("mid")
    val si3 = new StringIndexer().setInputCol("mccStr").setOutputCol("mcc")

    val pipeline = new Pipeline().setStages(Array(si1, si2, si3))
    val pipelineModel = pipeline.fit(dates)

    val finalDF = pipelineModel.transform(dates)
      .select("uid", "mid", "amount", "date", "mcc")
      .withColumn("uid", col("uid") + 1)
      .withColumn("mid", col("mid") + 1)

    //    val top50Merchant = finalDF.groupBy("mid").count().orderBy(col("count").desc)
    //    top50Merchant.show()

    //    finalDF.groupBy("uid").count().orderBy(col("count").desc).show(5000)

    finalDF
  }

}
