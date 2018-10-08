package com.intel.analytics.zoo.apps.movie

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

case class MovieData(userIdIndex: Double, itemIdIndex: Double, label: Double)

class DataPreprocess {

  def getDataDF() = {
    val mvpath = "/Users/guoqiong/intelWork/projects/wrapup/recommendation/ml-1m"

    val spark = SparkSession.builder().getOrCreate()
    import spark.sqlContext.implicits._

    val movielensDF = spark.sparkContext.textFile(mvpath + "/ratings.dat")
      .map(x => {
        val data: Array[Double] = x.split("::").map(n => n.toDouble)
        MovieData(data(0), data(1), data(2))
      })
      .toDF()

    // movielensDF.show()

    val minMaxRow = movielensDF.agg(min("userIdIndex"), max("userIdIndex"), min("itemIdIndex"), max("itemIdIndex")).collect()(0)

    val minUserId = minMaxRow.getDouble(0)
    val maxUserId = minMaxRow.getDouble(1)
    val minMovieId = minMaxRow.getDouble(2)
    val maxMovieId = minMaxRow.getDouble(3)
    val ratings = movielensDF.select("label").distinct().map(row => row.getDouble(0)).collect()
    println(minUserId + "," + maxUserId + "," + minMovieId + "," + maxMovieId + "," + ratings.mkString("|"))

    (movielensDF, maxUserId, maxMovieId)
  }

}
