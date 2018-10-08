package com.intel.analytics.zoo.apps.anomalyDetection

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, col, udf}


case class FeatureLabelIndex(feature: Array[Array[Float]], label: Float, index: Long) {
  override def toString =
    "value: " + feature.map(x => x.mkString("|")).mkString(",") + " label:" + label + " index:" + index
}

object Utils {


  def standardScaleHelper(df: DataFrame, colName: String) = {

    val mean = df.select(colName).agg(avg(col(colName))).collect()(0).getDouble(0)

    val stddevUdf = udf((num: Float) => (num - mean) * (num - mean))

    val stddev = Math.sqrt(df.withColumn("stddev", stddevUdf(col(colName)))
      .agg(avg(col("stddev"))).collect()(0).getDouble(0))

    println(colName + ",mean:" + mean + ",stddev:" + stddev)

    val scaledUdf = udf((num: Float) => (num - mean) / stddev)

    df.withColumn(colName, scaledUdf(col(colName)))

  }

  def standardScale(df: DataFrame, fields: Seq[String], index: Int = 0): DataFrame = {

    if (index == fields.length) {
      df
    } else {
      val colDf = standardScaleHelper(df, fields(index))
      standardScale(colDf, fields, index + 1)
    }
  }

  def distributeUnrollAll(dataRdd: RDD[Array[Float]], unrollLength: Int, predictStep: Int = 1): RDD[FeatureLabelIndex] = {

    val n = dataRdd.count()
    val indexRdd: RDD[(Array[Float], Long)] = dataRdd.zipWithIndex()

    //RDD[index of record, feature]
    val featureRdd: RDD[(Long, Array[Array[Float]])] = indexRdd
      .flatMap(x => {
        val pairs: Seq[(Long, List[(Array[Float], Long)])] = if (x._2 < unrollLength) {
          (0L to x._2).map(index => (index, List(x)))
        } else {
          (x._2 - unrollLength + 1 to x._2).map(index => (index, List(x)))
        }
        pairs
      }).reduceByKey(_ ++ _)
      .filter(x => x._2.size == unrollLength && x._1 <= n - unrollLength - predictStep)
      .map(x => {
        val data: Array[Array[Float]] = x._2.sortBy(y => y._2).map(x => x._1).toArray
        (x._1, data)
      }).sortBy(x => x._1)

    val skipIndex: Int = unrollLength - 1 + predictStep
    val labelRdd: RDD[(Long, Float)] = indexRdd.filter(x => x._2 >= skipIndex).map(x => (x._2 - skipIndex, x._1(0)))

    val featureData: RDD[FeatureLabelIndex] = featureRdd.join(labelRdd)
      .map(x => FeatureLabelIndex(x._2._1, x._2._2, x._1))

    featureData.cache()
  }

  def toSampleRdd(sc: SparkContext, x_train: Array[Array[Array[Float]]], y_train: Array[Float]) = {

    val train_data: Seq[(Array[Array[Float]], Float)] = x_train.zip(y_train).toSeq

    sc.parallelize(train_data, sc.defaultParallelism).map(x => {

      val shape: Array[Int] = Array(x._1.length, x._1(0).length)
      val data: Array[Float] = x._1.flatten
      val feature: Tensor[Float] = Tensor(data, shape)
      val label = Tensor[Float](T(x._2))
      Sample(feature, label)
    })

  }

  def toSampleRdd(sc: SparkContext, x_train: RDD[Array[Array[Float]]], y_train: RDD[Float]) = {

    val train_data = x_train.zip(y_train)

    train_data.map(x => {

      val shape: Array[Int] = Array(x._1.length, x._1(0).length)
      val data: Array[Float] = x._1.flatten
      val feature: Tensor[Float] = Tensor(data, shape)
      val label = Tensor[Float](T(x._2))
      Sample(feature, label)
    })

  }

  def toSampleRdd(sc: SparkContext, train_data: RDD[FeatureLabelIndex]) = {

    train_data.map(x => {

      val shape: Array[Int] = Array(x.feature.length, x.feature(0).length)
      val data: Array[Float] = x.feature.flatten
      val feature: Tensor[Float] = Tensor(data, shape)
      val label = Tensor[Float](T(x.label))
      Sample(feature, label)
    })

  }

  def distributeUnroll(dataRdd: RDD[(Array[Float], Long)], unrollLength: Int): RDD[Array[Array[Float]]] = {

    val n = dataRdd.count()

    dataRdd.map(x => x._1).zipWithIndex()
      .flatMap(x => {

        val pairs: Seq[(Long, List[(Array[Float], Long)])] = if (x._2 < unrollLength) {
          (0L to x._2).map(index => (index, List(x)))
        } else if (x._2 + unrollLength < n) {
          (x._2 - unrollLength + 1 to x._2).map(index => (index, List(x)))
        } else {
          Seq((-1, List(x)))
        }

        pairs
      })
      .reduceByKey(_ ++ _)
      .filter(x => x._1 != -1)
      .map(x => {

        val data: Array[Array[Float]] = x._2.sortBy(x => x._2).map(x => x._1).toArray
        (x._1, data)

      }).sortBy(x => x._1)
      .map(x => x._2)

  }
}
