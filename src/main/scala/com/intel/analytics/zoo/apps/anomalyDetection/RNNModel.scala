package com.intel.analytics.zoo.apps.anomalyDetection

import com.intel.analytics.bigdl.nn.keras.SimpleRNN
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Dropout, LSTM}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential


object RNNModel {

  def buildModel(inputShape: Shape): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(LSTM[Float](8, returnSequences = true, inputShape = inputShape))

    model.add(Dropout[Float](0.2))

    model.add(LSTM[Float](32, returnSequences = true))
    model.add(Dropout[Float](0.2))

    model.add(LSTM[Float](15, returnSequences = false))
    model.add(Dropout[Float](0.2))

    model.add(Dense[Float](outputDim = 1))

    model
  }

  def buildSimpleLSTM(inputShape: Shape): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(LSTM[Float](8, returnSequences = false, inputShape = inputShape))
    model
  }

  def buildSimpleRNN(inputShape: Shape): Sequential[Float] = {
    val model = Sequential[Float]()
    model.add(SimpleRNN[Float](8, returnSequences = false, inputShape = inputShape))
    model
  }

  def buildLSTM(inputShape: Shape): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(LSTM[Float](15, returnSequences = true, inputShape = Shape(50,1)))
    model.add(LSTM[Float](1, returnSequences = true)) //50

    model
  }

  //TODO
  def buildLSTMMul(inputShape: Shape): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(LSTM[Float](4, returnSequences = false, inputShape = Shape(50,4)))

    model
  }

}
