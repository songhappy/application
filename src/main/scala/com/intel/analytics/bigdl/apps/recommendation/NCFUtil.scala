package com.intel.analytics.bigdl.apps.recommendation

import org.apache.spark.sql.DataFrame
import scala.util.Random
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adam, LBFGS}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.ml.{DLClassifier, DLModel}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions._

object NCFUtil {

  def mixNegativeSamples(trainingData: DataFrame, userCount:Int, itemCount:Int): DataFrame = {
    import trainingData.sparkSession.implicits._

    val numRecords = (1.0 * trainingData.count()).toInt

    val ran = new Random(42L)
    val negativeSampleDF = (1 to numRecords).map { i =>
      val uid = Math.abs(ran.nextInt(userCount - 1)).toDouble + 1
      val mid = Math.abs(ran.nextInt(itemCount - 1)).toDouble + 1
      (uid, mid, 0L)
    }.distinct.toDF("userIdIndex", "itemIdIndex", "label")

    val removeDupDF = negativeSampleDF.join(trainingData, Seq("userIdIndex", "itemIdIndex"), "leftanti")

    val combineDF = trainingData.union(removeDupDF)
    require(combineDF.groupBy("userIdIndex", "itemIdIndex").count().filter("count > 1").count() == 0)
    combineDF
  }

  def getModel(userCount:Int, itemCount:Int): Module[Float] = {
    val userOutput = 20
    val itemOutput = 20
    val linear1 = 10

    val user_table = LookupTable(userCount, userOutput)
    val item_table = LookupTable(itemCount, itemOutput)

    user_table.setWeightsBias(Array(Tensor[Float](userCount, userOutput).randn(0, 0.1)))
    item_table.setWeightsBias(Array(Tensor[Float](itemCount, itemOutput).randn(0, 0.1)))

    val embedded_layer = Concat(2)
      .add(Sequential().add(Select(2,1)).add(user_table))
      .add(Sequential().add(Select(2,2)).add(item_table))
      .add(Sequential().add(Select(2,3)).add(View(1, 1)))

    val model = Sequential()
    model.add(embedded_layer)

    model.add(Linear(userOutput + itemOutput + 1, linear1)).add(ReLU())
    model.add(Linear(linear1, 2))
    model.add(LogSoftMax())

    return model
  }
}
