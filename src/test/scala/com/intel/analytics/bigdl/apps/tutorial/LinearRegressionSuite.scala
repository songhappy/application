package com.intel.analytics.bigdl.apps.tutorial

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FunSuite}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T, Table}
import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.bigdl.nn.{Linear, MSECriterion, Sequential}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.models.lenet.Utils._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{SGD, Top1Accuracy}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class LinearRegressionSuite extends FunSuite with BeforeAndAfter {

  var spark: SparkSession = _
  var sc: SparkContext = _
  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("Train")
      .setMaster("local[8]")
    spark = SparkSession.builder().config(conf).getOrCreate()
    sc = spark.sparkContext
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init
  }

  test("Linear") {
    val featuresDim = 2
    val dataLen = 100

    def GetRandSample() = {
      val features: Tensor[Float] = Tensor(featuresDim).rand(0, 1)
      val label = (0.4 + features.sum * 2).toFloat
      val sample = Sample[Float](features, label)
      sample
    }

    val rddTrain = sc.parallelize(0 until dataLen).map(_ => GetRandSample())


    // Parameters
    val learningRate = 0.2
    val trainingEpochs = 5
    val batchSize = 8
    val nInput = featuresDim
    val nOutput = 1

    def LinearRegression(nInput: Int, nOutput: Int) = {
      // Initialize a sequential container
      val model = Sequential()
      // Add a linear layer
      model.add(Linear(nInput, nOutput))
      model
    }

    val model = LinearRegression(nInput, nOutput)

    val optimizer = Optimizer(model = model,
      sampleRDD = rddTrain,
      criterion = MSECriterion[Float](),
      batchSize = batchSize)
    optimizer.setOptimMethod(new SGD(learningRate = learningRate))
    optimizer.setEndWhen(Trigger.maxEpoch(trainingEpochs))

    val trainedModel = optimizer.optimize()

    val predictResult: RDD[Activity] = trainedModel.predict(rddTrain)

    val p = predictResult.take(5).map(_.toTensor.valueAt(1)).mkString(",")
    println("Predict result:")
    println(p)

    val r = new scala.util.Random(100)
    val totalLength = 10
    val features = Tensor(totalLength, featuresDim).rand(0, 1)
    var label = (0.4 + features.sum).toFloat
    val prediction = sc.parallelize(0 until totalLength).map(r => Sample[Float](features(r + 1), label))
    val predictResult2 = trainedModel.predict(prediction)
    val p2 = predictResult2.take(6).map(_.toTensor.valueAt(1))
    val groundLabel = Tensor(T(
      T(-0.47596836f),
      T(-0.37598032f),
      T(-0.00492062f),
      T(-0.5906958f),
      T(-0.12307882f),
      T(-0.77907401f)))

    var mse = 0f
    for (i <- 1 to 6) {
      mse += (p(i - 1) - groundLabel(i).valueAt(1)) * (p2(i - 1) - groundLabel(i).valueAt(1))
    }
    mse /= 6f
    println(mse)

  }

  test("df liner") {

  }

}
