package com.intel.analytics.bigdl.apps.tutorial

import org.scalatest.{BeforeAndAfter, FunSuite}
import com.intel.analytics.bigdl.nn.JoinTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


class understandLayers extends FunSuite with BeforeAndAfter {

  var sc: SparkContext = _
  var spark: SparkSession = _

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("TrainWithNCF")
      .setMaster("local[8]")
    sc = new SparkContext(conf)
    spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init
  }

  test("JoinTable") {

    val layer = JoinTable(3, 0)
    val input1 = Tensor(T(T(
      T(1f, 2f, 3f, 0f),
      T(3f, 4f, 5f, 0f)))
    )

    val input2 = Tensor(T(T(
      T(3f, 4f, 5f, 0f),
      T(1f, 2f, 3f, 0f)))
    )

    val input = T(input1, input2)

    val gradOutput = Tensor(T(
      T(1f, 2f, 3f, 3f, 4f, 5f, 0f),
      T(3f, 4f, 5f, 1f, 2f, 3f, 0f)
    ))

    val output = layer.forward(input)
   // val grad = layer.backward(input, gradOutput)

    println(output)
    println(input1.nDimension())
    println(input1.size().toList)

  }


  test("linear") {


  }




  test("simple model") {

    import com.intel.analytics.bigdl.dataset._
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.optim._
    import com.intel.analytics.bigdl.tensor.Tensor
    import com.intel.analytics.bigdl.utils.T

    // Define the model
    val model = Linear[Float](2, 1)
    model.bias.zero()

    // Generate 2D dummy data, y = 0.1 * x[1] + 0.3 * x[2]
    val samples = Seq(
      Sample[Float](Tensor[Float](T(5f, 5f)), Tensor[Float](T(2.0f))),
      Sample[Float](Tensor[Float](T(-5f, -5f)), Tensor[Float](T(-2.0f))),
      Sample[Float](Tensor[Float](T(-2f, 5f)), Tensor[Float](T(1.3f))),
      Sample[Float](Tensor[Float](T(-5f, 2f)), Tensor[Float](T(0.1f))),
      Sample[Float](Tensor[Float](T(5f, -2f)), Tensor[Float](T(-0.1f))),
      Sample[Float](Tensor[Float](T(2f, -5f)), Tensor[Float](T(-1.3f)))
    )
    val trainData: RDD[Sample[Float]] = sc.parallelize(samples, 1)

    // Define the model
    val optimizer = Optimizer[Float](model, trainData, MSECriterion[Float](), 4)

    optimizer.optimize()

    println("--------------------------------")
    println(model.weight)
  }

}
