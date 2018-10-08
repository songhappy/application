package com.intel.analytics.zoo.apps.anomalyDetection

import com.intel.analytics.zoo.apps.anomalyDetection.Utils.{distributeUnroll, standardScale}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

class tmp {

  def prepareData3(df: DataFrame, size: Int) = {

    val scaledDF = standardScale(df, Seq("value", "hour", "awake"))
    val dataRdd: RDD[(Array[Float], Long)] = scaledDF.rdd.map(row => Array
    (row.getAs[Double](0).toFloat, row.getAs[Double](1).toFloat, row.getAs[Double](2).toFloat))
      .zipWithIndex()

    val prediction_time = 1
    val testdataSize = 1000
    val unroll_length = 50
    val testdatacut = testdataSize + unroll_length + 1
    val n = dataRdd.count()

    val x_train: RDD[(Array[Float], Long)] = dataRdd.filter(x => x._2 < n - prediction_time - testdatacut)
    val y_train = dataRdd.filter(x => x._2 > prediction_time && x._2 <= n - testdatacut)

    val x_test: RDD[(Array[Float], Long)] = dataRdd.filter(x => x._2 >= n - testdatacut && x._2 < n - prediction_time)
    val y_test = dataRdd.filter(x => x._2 >= n - (testdatacut - prediction_time) && x._2 <= n)

    println("after train test split")
    println(x_train.count + "," + y_train.count() + "," + x_test.count() + "," + y_test.count())

    val x_train_unroll: RDD[Array[Array[Float]]] = distributeUnroll(x_train, unroll_length)
    val x_test_unroll: RDD[Array[Array[Float]]] = distributeUnroll(x_test, unroll_length)

    val y_train_length = y_train.count()
    val x_train_sliding_length = x_train_unroll.count()
    val y_test_length = y_test.count()
    val x_test_sliding_length = x_test_unroll.count()

    val y_train_t: RDD[Float] = y_train.map(x => x._1).zipWithIndex().filter(x => x._2 >= y_train_length - x_train_sliding_length && x._2 <= y_train_length).map(x => x._1(0))
    val y_test_t: RDD[Float] = y_test.map(x => x._1).zipWithIndex().filter(x => x._2 >= y_test_length - x_test_sliding_length && x._2 < y_test_length).map(x => x._1(0))

    println("after unroll data")
    println(x_train_sliding_length + "," + y_train_t.count() + "," + x_test_sliding_length + "," + y_test_t.count())

    (x_train_unroll, y_train_t, x_test_unroll, y_test_t)

  }


  def prepareData2(df: DataFrame, size: Int) = {

    val normalizeData = standardScale(df, Seq("value", "hour", "awake"))

    normalizeData.show(20, false)

    val data_n: Array[Array[Float]] = normalizeData.rdd.map(row => Array(row.getAs[Double](0).toFloat
      , row.getAs[Double](1).toFloat, row.getAs[Double](2).toFloat)).collect()

    val prediction_time = 1
    val testdatasize = 1000
    val unroll_length = 50
    val testdatacut = testdatasize + unroll_length + 1
    val n = data_n.length
    //train data
    val x_train: Array[Array[Float]] = data_n.slice(0, n - prediction_time - testdatacut)
    val y_train: Array[Float] = data_n.map(x => x(0)).slice(prediction_time, n - testdatacut)

    // test data
    val x_test = data_n.slice(n - testdatacut, n - prediction_time)
    val y_test: Array[Float] = data_n.map(x => x(0)).slice(n - (testdatacut - prediction_time), n)

    println(x_train.length + "," + y_train.length + "," + x_test.length + "," + y_test.length)

    val x_train_sliding = x_train.sliding(unroll_length).toArray
    val x_test_sliding = x_test.sliding(unroll_length).toArray

    println(y_train.length + "," + x_train_sliding.length + "," + y_test.length + "," + x_test_sliding.length)

    val y_train_t: Array[Float] = y_train.slice(y_train.length - x_train_sliding.length, y_train.length)

    val y_test_t = y_test.slice(y_test.length - x_test_sliding.length, y_test.length)

    println(x_train_sliding.length + "," + y_train_t.length + "," + x_test_sliding.length + "," + y_test_t.length)

    (x_train_sliding, y_train_t, x_test_sliding, y_test_t)
  }


}
