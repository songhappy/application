package com.intel.analytics.zoo.apps.tutorial

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.keras.{Dense, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{Engine, Shape}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FunSuite}

class RnnLayers extends FunSuite with BeforeAndAfter {

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

  test("Simple RNN layer") {

    val hiddenSize = 4
    val inputSize = 5
    val module = Recurrent().add(RnnCell(inputSize, hiddenSize, Tanh()))
    val input = Tensor(Array(1, 5, inputSize))
    for (i <- 1 to 5) {
      val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0f)
    }

    val output = module.forward(input)

    println(input.size().mkString("|"))
    println(input)
    println(output.size().mkString("|"))
    println(output)
    val state = module.getHiddenState()
    module.setHiddenState(state)


  }

  test("SimpleRNN model") {
    def buildModel(inputSize: Int, hiddenSize: Int, outputSize: Int) = {
      val model = Sequential[Float]()
      model.add(Recurrent[Float]()
        .add(RnnCell[Float](inputSize, hiddenSize, Tanh[Float]())))
        .add(TimeDistributed[Float](Linear[Float](hiddenSize, outputSize)))
      model
    }
  }

  test("dense") {
    val model = Sequential[Float]()
    model.add(Dense(1, activation = "relu", inputShape = Shape(4)))
    val input = Tensor[Float](2, 4).randn()
    val output = model.forward(input)
    println(output.toTensor.size().mkString("|"))
  }


  test("flatten") {
    val x = Array(Array(0, 1, 2), Array(1, 2, 3))
    println(x.toList)
    println(x.flatten.toList)
  }

  test("RNN Models") {
    val model = com.intel.analytics.zoo.apps.anomalyDetection.RNNModel.buildSimpleRNN(inputShape = Shape(4, 5))
    val input = Tensor[Float](1, 4, 5)
    val d1l = 1
    val d2l = 4
    val d3l = 5
    val weight = Tensor(d1l, d2l, d3l)
    val page = d2l * d3l
    for (i <- 0 to d1l * d2l * d3l - 1) {
      val d1 = i / page + 1
      val d2 = (i % page) / (d3l) + 1
      val d3 = (i % page) % d3l + 1
      weight.setValue(d1, d2, d3, i)
    }
    val output = model.forward(weight)
    println(output)

    println(model.parameters()._1.toList)

  }
}
