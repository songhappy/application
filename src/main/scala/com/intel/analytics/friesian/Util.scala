package com.intel.analytics.friesian

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable.WrappedArray
import scala.util.Random

case class GroupedIndex(asin_index: Long, cat_index: Long, unixReviewTime: Long, asin_history: Array[Long], cat_history: Array[Long])

case class Negative(item: Long, label: Int)

object Util {

  def createCategoriesMap(metaBooks: DataFrame): Map[String, Long] = {

    val map = metaBooks.select("categories")
      .withColumn("flatten_cat", explode(col("categories")))
      .drop("categories")
      .withColumn("flatten_cat", explode(col("flatten_cat")))
      .distinct.rdd.map(r => r.getAs[String](0))
      .collect()
      .zipWithIndex.map(x => (x._1, (x._2 + 1).toLong)).toMap
    print(map.take(10))
    map
  }

  def indexCategories(categoriesMap: Map[String, Long], metaBooks: DataFrame): DataFrame = {

    def lookupCatUdf = {
      val func = (categories: WrappedArray[WrappedArray[String]]) => // need more logic, only get the last one here
        if (categories == null) 0L else categoriesMap(categories(0).last)
      udf(func)
    }


    metaBooks.select("asin", "categories")
      .withColumn("categories_index", lookupCatUdf(col("categories")))
      .drop("categories")

  }

  private def indexColumn(df: DataFrame, columnName: String): DataFrame = {

    import df.sparkSession.implicits._
    df.select(columnName).distinct().rdd.map(row => row.getAs[String](0))
      .zipWithIndex().toDF(columnName, columnName + "_index")
  }

  def join(reviews: DataFrame, metaBooks: DataFrame): DataFrame = {
    val categoriesMap = createCategoriesMap(metaBooks)

    val booksWithCategories = indexCategories(categoriesMap, metaBooks)
    booksWithCategories.show(10, false)
    val joinDF = reviews.select("asin", "overall", "reviewerID", "unixReviewTime")
      .join(booksWithCategories, Seq("asin"), "inner")

    joinDF.show(10, false)

    val items = indexColumn(joinDF, "asin")
    val reviewers = indexColumn(joinDF, "reviewerID")

    joinDF.join(items, Seq("asin"), "inner")
      .join(reviewers, Seq("reviewerID"), "inner")
      .drop("asin")
      .drop("reviewerID")

  }

  def createHistorySeq(df: DataFrame, maxLength: Int): DataFrame = {

    val asinUdf = udf(f = (asin_collect: Seq[Row]) => {
      val full_rows = asin_collect.sortBy(x => x.getAs[Long](2)).toArray

      val n = full_rows.length

      val range: Seq[Int] = if (maxLength < n) {
        (n - maxLength to n - 1)
      } else {
        (0 to n - 1)
      }

      range.map(x =>
        GroupedIndex(asin_index = full_rows(x).getAs[Long](0),
          cat_index = full_rows(x).getAs[Long](1),
          unixReviewTime = full_rows(x).getAs[Long](2),
          asin_history = full_rows.slice(0, x).map(row => row.getAs[Long](0)),
          cat_history = full_rows.slice(0, x).map(row => row.getAs[Long](1))))
    })

    val aggDF = df.groupBy("reviewerID_index")
      .agg(collect_list(struct(col("asin_index"), col("categories_index"), col("unixReviewTime"))).as("asin_collect"))

    aggDF.withColumn("item_history", asinUdf(col("asin_collect")))
      .withColumn("item_history", explode(col("item_history")))
      .drop("asin_collect")
      .select(col("reviewerID_index"),
        col("item_history.asin_index").as("asin_index"),
        col("item_history.cat_index").as("cat_index"),
        col("item_history.unixReviewTime").as("unixReviewTime"),
        col("item_history.asin_history").as("asin_history"),
        col("item_history.cat_history").as("cat_history"))
      .filter("size(asin_history) > 0 and size(cat_history) > 0")

  }


  def prepad(df: DataFrame, colNames: Array[String], maxLength: Int = 10, index: Int = 0): DataFrame = {

    if (index == colNames.length) {
      df
    } else {

      val padUdf = udf((history: WrappedArray[Long]) => {
        val n = history.length
        if (maxLength > n) {
          ((0 to maxLength - n).map(_ => 0L) ++ history)
        } else {
          history.slice(n - maxLength, n)
        }

      })

      val paddedDf = df.withColumn(colNames(index) + "_padded", padUdf(col(colNames(index))))
      prepad(paddedDf, colNames, maxLength, index + 1)
    }

  }


  def mask(df: DataFrame, colNames: Array[String], index: Int = 0): DataFrame = {

    if (index == colNames.length) {
      df
    } else {

      val maskUdf = udf((history: WrappedArray[Long]) => history.map(x => if (x > 0) 1 else 0))
      val maskDF = df.withColumn(colNames(index) + "_mask", maskUdf(col(colNames(index) + "_padded")))
      mask(maskDF, colNames, index + 1)
    }
  }

  def addNegSampling(df: DataFrame, itemSize: Int, userID: String = "reviewerID_index", itemID: String = "asin_index",
                     label: String = "label", negNum: Int = 1): DataFrame = {
    val sqlContext = df.sqlContext
    val colNames = df.columns
    val restCols = colNames.filter(!_.contains(itemID))
    val combinedRDD = df.rdd.flatMap(row => {
      val restValues = row.getValuesMap[Any](restCols).values
      val result = new Array[Row](negNum + 1)
      val r = new Random()
      for (i <- 0 until negNum) {
        var neg = 0
        do {
          neg = r.nextInt(itemSize)
        } while (neg == row.getAs[Long](itemID))

        result(i) = Row.fromSeq(restValues.toSeq ++ Array[Any](neg, 0))
      }
      result(negNum) = Row.fromSeq(restValues.toSeq ++ Array[Any](row.getAs(itemID), 1))
      result

    })
    val newSchema = StructType(df.schema.fields.filter(_.name != itemID) ++ Array(
      StructField(itemID, IntegerType, false), StructField(label, IntegerType, false)))

    val combinedDF = sqlContext.createDataFrame(combinedRDD, newSchema)
    combinedDF
  }

  def addNegSamplingUdf(df: DataFrame, itemSize: Int, itemID: String = "asin_index", negNum: Int = 1): DataFrame = {

    val r = new Random()

    val negativeUdf = udf((itemID: Long) => (1 to negNum).map(x => {
      var neg = 0
      do {
        neg = r.nextInt(itemSize)
      } while (neg == itemID)
      neg
    }).map(x => Negative(x, 0)) ++ Seq(Negative(itemID, 1)))

    val columns = df.columns.filter(x => x != itemID).mkString(",")

    val negativedf = df.withColumn("negative", negativeUdf(col(itemID)))
      .withColumn("negative", explode(col("negative")))

    negativedf.createOrReplaceTempView("tmp")

    df.sqlContext.sql(s"select $columns , negative.item as itemID, negative.label as label from tmp")

  }

  def addNegHistorySequence(df: DataFrame, itemSize: Int, itemCategoryMap: Map[Long, Long], negNum: Int = 1): DataFrame = {
    val sqlContext = df.sqlContext
    val combinedRDD = df.rdd.map(row => {
      val item_history = row.getAs[WrappedArray[Long]]("asin_history")
      val r = new Random()
      val negItemSeq = Array.ofDim[Long](item_history.length, negNum)
      val negCatSeq = Array.ofDim[Long](item_history.length, negNum)
      for (i <- 0 until item_history.length) {
        for (j <- 0 until negNum) {
          var negItem = 0
          do {
            negItem = r.nextInt(itemSize)
          } while (negItem == item_history(i))
          negItemSeq(i)(j) = negItem
          negCatSeq(i)(j) = itemCategoryMap(negItem)
        }
      }

      val result = Row.fromSeq(row.toSeq ++ Array[Any](negItemSeq, negCatSeq))
      result

    })
    val newSchema = StructType(df.schema.fields ++ Array(
      StructField("noclk_item_list", ArrayType(ArrayType(LongType))),
      StructField("noclk_cat_list", ArrayType(ArrayType(LongType)))))

    val combinedDF = sqlContext.createDataFrame(combinedRDD, newSchema)
    combinedDF
  }


}
