package com.intel.analytics.bigdl.apps.anomalyDetection

import com.intel.analytics.bigdl.dataset.{DataSet, FixedLength, PaddingParam, SampleToMiniBatch}
import com.intel.analytics.bigdl.dataset.text.{Dictionary, LabeledSentence, LabeledSentenceToSample, TextToLabeledSentence}
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.models.rnn.SequencePreprocess
import com.intel.analytics.bigdl.models.rnn.Test.logger
import com.intel.analytics.bigdl.models.rnn.Utils.{TestParams, TrainParams, readSentence}
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim.{Loss, ValidationMethod}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FunSuite}

class RnnTestSuite extends FunSuite with BeforeAndAfter{

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

  test("test"){

//    case class TestParams(
//                           folder: String = "./",
//                           modelSnapshot: Option[String] = None,
//                           numOfWords: Option[Int] = Some(10),
//                           evaluate: Boolean = true,
//                           sentFile: Option[String] = None,
//                           tokenFile: Option[String] = None,
//                           batchSize: Int = 4
//                         )

    val dataPath = "/Users/guoqiong/intelWork/projects/wrapup/rnn"

    val input = dataPath + "/inputdata"
    val sentFile = dataPath + "/bin/en-sent.bin"
    val tokenFile = dataPath + "/bin/en-token.bin"

    val param = new TestParams(folder = dataPath +"/saveDict",
      modelSnapshot = Some(dataPath +"/model/20171116_223721/model.34861"),
      batchSize = 4,
      sentFile = Some(sentFile),
      tokenFile = Some(tokenFile)
    )

    val vocab = Dictionary(param.folder)
    val model = Module.load[Float](param.modelSnapshot.get)

    if (param.evaluate) {
      val valtokens = SequencePreprocess(
        param.folder + "/test.txt",
        sc = sc,
        param.sentFile,
        param.tokenFile).collect()
      val maxValLength = valtokens.map(x => x.length).max

      val totalVocabLength = vocab.getVocabSize() + 1
      val startIdx = vocab.getIndex(SentenceToken.start)
      val endIdx = vocab.getIndex(SentenceToken.end)
      val padFeature = Tensor[Float]().resize(totalVocabLength)
      padFeature.setValue(endIdx + 1, 1.0f)
      val padLabel = Tensor[Float](T(startIdx.toFloat + 1.0f))
      val featurePadding = PaddingParam(Some(Array(padFeature)),
        FixedLength(Array(maxValLength)))
      val labelPadding = PaddingParam(Some(Array(padLabel)),
        FixedLength(Array(maxValLength)))

      val evaluationSet = DataSet.array(valtokens)
        .transform(TextToLabeledSentence[Float](vocab))
        .transform(LabeledSentenceToSample[Float](totalVocabLength))
        .transform(SampleToMiniBatch[Float](param.batchSize,
          Some(featurePadding), Some(labelPadding))).toLocal()

      val result = model.evaluate(evaluationSet,
        Array(new Loss[Float](
          TimeDistributedCriterion[Float](
            CrossEntropyCriterion[Float](),
            sizeAverage = true)).asInstanceOf[ValidationMethod[Float]]))

      result.foreach(r => println(s"${r._2} is ${r._1}"))
    } else {
      val timeDim = 2
      val featDim = 3
      val concat = Tensor[Float]()
      val lines = readSentence(param.folder)
      val input = lines.map(x =>
        x.map(t => vocab.getIndex(t).toFloat))
      val labeledInput = input.map(x =>
        new LabeledSentence[Float](x, x))

      val vocabSize = vocab.getVocabSize() + 1
      val batchSize = param.batchSize

      val rdd = sc.parallelize(labeledInput).mapPartitions(iter =>
        LabeledSentenceToSample[Float](vocabSize).apply(iter)
      ).mapPartitions(iter =>
        SampleToMiniBatch[Float](batchSize).apply(iter)
      )

      val flow = rdd.mapPartitions(iter => {
        iter.map(batch => {
          var curInput = batch.getInput().toTensor[Float]
          // Iteratively output predicted words
          for (i <- 1 to param.numOfWords.getOrElse(0)) {
            val input = curInput.max(featDim)._2
            val output = model.forward(curInput).toTensor[Float]
            val predict = output.max(featDim)._2.select(timeDim, output.size(timeDim))
            concat.resize(curInput.size(1), curInput.size(timeDim) + 1, curInput.size(featDim))
            concat.narrow(timeDim, 1, curInput.size(timeDim)).copy(curInput)
            for (j <- 1 to curInput.size(1)) {
              concat.setValue(j, concat.size(timeDim), predict.valueAt(j, 1).toInt + 1, 1.0f)
            }
            curInput = concat
          }
          val predIdx = curInput.max(featDim)._2
          val predArray = new Array[Float](predIdx.nElement())
          Array.copy(predIdx.storage().array(), predIdx.storageOffset() - 1,
            predArray, 0, predIdx.nElement())
          predArray.grouped(predIdx.size(timeDim)).toArray[Array[Float]]
        })
      }).collect().flatMap(x => x)

      val results = flow.map(x => x.map(t => vocab.getWord(t)))
      results.foreach(x => logger.info(x.mkString(" ")))
    }
  }
}
