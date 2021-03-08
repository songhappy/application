package com.intel.analytics.friesian

import org.apache.spark.sql.{SQLContext, SaveMode, SparkSession}
import scopt.OptionParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import Util._
import org.apache.spark.sql.functions._
import org.joda.time.DateTime

case class PreprocessParms(
                            val inputDir: String = "/Users/guoqiong/intelWork/git/friesian/data/book_review/",
                            val outputDir: String = "users/guoqiong/intelWork/git/friesian/data/preprocessed/",
                            val hisLenMin: Int = 1,
                            val hisLenMax: Int = 10,
                            val negativeNum: Int = 1,
                            val nonClk: Int = 5
                          )

object Preprocessing {


  def main(args: Array[String]): Unit = {

    val preprocessParms = PreprocessParms()

    val parser = new OptionParser[PreprocessParms]("DLRM Preprocessing") {
      opt[String]("input")
        .text(s"input")
        //.required()
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("outputDir")
        .text(s"outputDir")
        //.required()
        .action((x, c) => c.copy(outputDir = x))
      opt[Int]('l', "hisLenMax")
        .text(s"hisLenMax")
        .action((x, c) => c.copy(hisLenMax = x))
      opt[Int]('n', "nonClk")
        .text(s"none click for each item, default is 5")
        .action((x, c) => c.copy(nonClk = x))

    }

    parser.parse(args, preprocessParms).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(parms: PreprocessParms): Unit = {
    val begin =  DateTime.now()
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Preprocess for DLRM")
    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    val metaBooks = spark.read.json(parms.inputDir + "meta_Books.json")
    val reviews = spark.read.json(parms.inputDir + "point02reviews.json")

    val joined = Util.join(reviews, metaBooks)

    val categoryMap = joined.select("asin_index", "categories_index").distinct()
      .rdd.map(row => (row.getAs[Long](0), row.getAs[Long](1)))
      .collect().toMap

    val withHistory = Util.createHistorySeq(joined, 100)

    val withNegativeSamples = Util.addNegSamplingUdf(withHistory, categoryMap.size)


    val withNegativeHistory = Util.addNegHistorySequence(withNegativeSamples, categoryMap.size, categoryMap)

    val prepadDF = Util.prepad(withNegativeHistory, Array("cat_history", "asin_history"))
    val maskDF = Util.mask(prepadDF, Array("cat_history", "asin_history"))

    maskDF.show(10, false)
    maskDF.write.mode(SaveMode.Overwrite).parquet(parms.outputDir+"data")
    val end =  DateTime.now()
    print((end.getMillis()-begin.getMillis())/1000d, "second")
  }

}
