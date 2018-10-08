package com.intel.analytics.bigdl.apps.anomalyDetection

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, FixedLength, PaddingParam, SampleToMiniBatch}
import com.intel.analytics.bigdl.dataset.text.LabeledSentenceToSample
import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.models.rnn.{SequencePreprocess, SimpleRNN}
import com.intel.analytics.bigdl.models.rnn.Utils._
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FunSuite}

class RnnTrainSuite extends FunSuite with BeforeAndAfter {

  var spark: SparkSession = _
  var sc: SparkContext = _
  val logger = Logger.getLogger(getClass)

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("Train")
      .setMaster("local[8]")
    spark = SparkSession.builder().config(conf).getOrCreate()
    sc = spark.sparkContext
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init
  }

  test("train") {

//    case class TrainParams(
//                            dataFolder: String = "./",
//                            saveFolder: String = "./",
//                            modelSnapshot: Option[String] = None,
//                            stateSnapshot: Option[String] = None,
//                            checkpoint: Option[String] = None,
//                            batchSize: Int = 128,
//                            learningRate: Double = 0.1,
//                            momentum: Double = 0.0,
//                            weightDecay: Double = 0.0,
//                            dampening: Double = 0.0,
//                            hiddenSize: Int = 40,
//                            vocabSize: Int = 4000,
//                            bptt: Int = 4,
//                            nEpochs: Int = 30,
//                            sentFile: Option[String] = None,
//                            tokenFile: Option[String] = None,
//                            overWriteCheckpoint: Boolean = false)

    val dataPath = "/Users/guoqiong/intelWork/projects/wrapup/rnn"



    val input = dataPath + "/inputdata"
    val sentFile = dataPath + "/bin/en-sent.bin"
    val tokenFile = dataPath + "/bin/en-token.bin"

    val tokens = SequencePreprocess(
      input + "/train.txt",
      sc = sc,
      Option(sentFile),
      Option(tokenFile))

    val param = new TrainParams(dataFolder = dataPath +"/inputdata",
      saveFolder = dataPath +"/saveDict",
      checkpoint = Some(dataPath+"/model"),
      batchSize = 24,
      sentFile = Some(sentFile),
      tokenFile = Some(tokenFile)
    )
    val dictionary = Dictionary(tokens, param.vocabSize)
    dictionary.save(param.saveFolder)

    val maxTrainLength = tokens.map(x => x.length).max

    val valtokens = SequencePreprocess(
      param.dataFolder + "/val.txt",
      sc = sc,
      param.sentFile,
      param.tokenFile)
    val maxValLength = valtokens.map(x => x.length).max

    logger.info(s"maxTrain length = ${maxTrainLength}, maxVal = ${maxValLength}")

    val totalVocabLength = dictionary.getVocabSize() + 1
    val startIdx = dictionary.getIndex(SentenceToken.start)
    val endIdx = dictionary.getIndex(SentenceToken.end)
    val padFeature = Tensor[Float]().resize(totalVocabLength)
    padFeature.setValue(endIdx + 1, 1.0f)
    val padLabel = Tensor[Float](T(startIdx.toFloat + 1.0f))
    val featurePadding = PaddingParam(Some(Array(padFeature)),
      FixedLength(Array(maxTrainLength)))
    val labelPadding = PaddingParam(Some(Array(padLabel)),
      FixedLength(Array(maxTrainLength)))

    val trainSet = DataSet.rdd(tokens)
      .transform(TextToLabeledSentence[Float](dictionary))
      .transform(LabeledSentenceToSample[Float](totalVocabLength))
      .transform(SampleToMiniBatch[Float](
        param.batchSize,
        Some(featurePadding),
        Some(labelPadding)))

    val validationSet = DataSet.rdd(valtokens)
      .transform(TextToLabeledSentence[Float](dictionary))
      .transform(LabeledSentenceToSample[Float](totalVocabLength))
      .transform(SampleToMiniBatch[Float](param.batchSize,
        Some(featurePadding), Some(labelPadding)))

    val model = if (param.modelSnapshot.isDefined) {
      Module.load[Float](param.modelSnapshot.get)
    } else {
      val curModel = SimpleRNN(
        inputSize = totalVocabLength,
        hiddenSize = param.hiddenSize,
        outputSize = totalVocabLength)
      curModel.reset()
      curModel
    }

    val optimMethod = if (param.stateSnapshot.isDefined) {
      OptimMethod.load[Float](param.stateSnapshot.get)
    } else {
      new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
        weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening)
    }

    val optimizer = Optimizer(
      model = model,
      dataset = trainSet,
      criterion = TimeDistributedCriterion[Float](
        CrossEntropyCriterion[Float](), sizeAverage = true)
    )

    if (param.checkpoint.isDefined) {
      optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
    }

    if(param.overWriteCheckpoint) {
      optimizer.overWriteCheckpoint()
    }

    optimizer.setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float](
        TimeDistributedCriterion[Float](CrossEntropyCriterion[Float](), sizeAverage = true))))
      .setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(param.nEpochs))
      .setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      .optimize()
    sc.stop()

  }

}