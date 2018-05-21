package com.intel.analytics.bigdl.apps.recommendation

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.ml.feature.{LabeledPoint, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

import scala.collection.mutable
import scala.util.Random

object Utils {

  val add1 = udf((num: Double) => num + 1)

  val sizeofArray = udf((feature: mutable.WrappedArray[Double]) => feature.length)

  def addNegativeSample1(indexedDF: DataFrame) = {

    val row = indexedDF.agg(max("userIdIndex"), max("itemIdIndex")).head
    val (userCount, itemCount) = (row.getAs[Double](0).toInt, row.getAs[Double](1).toInt)

    println(userCount + "," + itemCount)
    val sampleDict = indexedDF.rdd.map(row => row(0) + "," + row(1)).collect().toSet

    val numberRecords = 1 * indexedDF.count

    import indexedDF.sparkSession.implicits._

    val ran = new Random(seed = 42L)
    val negativeSampleDF = indexedDF.sparkSession.sparkContext
      .parallelize(0 to numberRecords.toInt)
      .map(x => {
        val uid = Math.max(ran.nextInt(userCount), 1)
        val iid = Math.max(ran.nextInt(itemCount), 1)
        (uid, iid)
      })
      .filter(x => !sampleDict.contains(x._1 + "," + x._2)).distinct()
      .map(x => (x._1, x._2, 0.0))
      .toDF("userIdIndex", "itemIdIndex", "label")

    indexedDF.union(negativeSampleDF)
  }

  def getNegativeSamples(amplifier: Int, trainDF: DataFrame) = {

    val indexedDF = trainDF.select("userIdIndex", "itemIdIndex", "label")
    val row = indexedDF.agg(max("userIdIndex"), max("itemIdIndex")).head
    val (userCount, itemCount) = (row.getAs[Double](0).toInt, row.getAs[Double](1).toInt)

    println(userCount + "," + itemCount)
    val sampleDict = indexedDF.rdd.map(row => row(0) + "," + row(1)).collect().toSet

    val numberRecords = amplifier * indexedDF.count

    import indexedDF.sparkSession.implicits._

    val ran = new Random(seed = 42L)
    val negativeSampleDF = indexedDF.sparkSession.sparkContext
      .parallelize(0 to numberRecords.toInt)
      .map(x => {
        val uid = Math.max(Random.nextInt(userCount), 1)
        val iid = Math.max(Random.nextInt(itemCount), 1)
        (uid, iid)
      })
      .filter(x => !sampleDict.contains(x._1 + "," + x._2)).distinct()
      .map(x => (x._1, x._2, 0.0))
      .toDF("userIdIndex", "itemIdIndex", "label")

    negativeSampleDF
  }

  def df2rddOfSample(indexed: DataFrame): RDD[Sample[Float]] = {

    val rddOfSample = indexed.rdd.map(row => {
      val uid = row.getAs[Double](0).toFloat
      val iid = row.getAs[Double](1).toFloat
      val label = row.getAs[Double](2).toFloat

      val feature: Tensor[Float] = Tensor(T(uid, iid))
      Sample(feature, label)
    })
    rddOfSample
  }

  val df2LP: (DataFrame) => DataFrame = df => {
    import df.sparkSession.implicits._
    df.select("userIdIndex", "itemIdIndex", "label").rdd.map { r =>
      val f: linalg.Vector = Vectors.dense(r.getDouble(0), r.getDouble(1))
      require(f.toArray.take(2).forall(_ >= 0))
      val l = r.getDouble(2)
      LabeledPoint(l, f)
    }.toDF().orderBy(rand()).cache()
  }

  val df2LP2: (DataFrame) => DataFrame = df => {
    import df.sparkSession.implicits._
    df.select("userVec", "itemVec", "label").rdd.map { r =>
      val vec = r.getSeq[Float](0) ++ r.getSeq[Float](1)
      val vect: Seq[Double] = vec.map(_.toDouble)
      val f: linalg.Vector = Vectors.dense(vect.toArray)
      require(f.toArray.size == 100)
      val l = r.getDouble(2)
      LabeledPoint(l, f)
    }.toDF().orderBy(rand()).cache()
  }


  def getFeaturesLP(trainDF: DataFrame): DataFrame = {

    val featurizerUdf = udf((userVec: mutable.WrappedArray[Float],
                             itemVec: mutable.WrappedArray[Float]) => {

      val vec = userVec ++ itemVec
      val vect: Seq[Double] = vec.map(_.toDouble)
      val f: linalg.Vector = Vectors.dense(vect.toArray)
      require(f.toArray.size == 100)
      f.toArray
    })

    trainDF.withColumn("features", featurizerUdf(col("userVec"), col("itemVec")))
  }

  val toZero = udf { d: Double =>
    if (d > 1) 1.0 else 0.0
  }

  def getCosineSim = {
    val func =
      (arg1: mutable.WrappedArray[Float], arg2: mutable.WrappedArray[Float]) =>
        math.abs(Similarity.cosineSimilarity(arg1.toArray, arg2.toArray))
    udf(func)
  }

  val sizeFilter = {
    val func = (arg: mutable.WrappedArray[Any]) =>
      if ((arg.toArray.length) > 1) true else false
    udf(func)
  }

  def score2label(cutPoint: Float) = {
    val func = (arg: Float) => if (arg > cutPoint) 1.0 else 0.0
    udf(func)
  }

  def calculatePreRec(cutPoint: Float, dfin: DataFrame) = {
    val df = dfin.withColumn("prediction", score2label(cutPoint)(col("score")))
    val groundPositive = df.filter(col("label") === 1.0).count()
    val predictedPositive = df.filter(col("prediction") === 1.0).count()
    val truePositive = df.filter(col("label") === 1.0 && col("prediction") === 1.0).count().toFloat
    val precision = truePositive / predictedPositive
    val recall = truePositive / groundPositive
    (cutPoint, precision, recall)
  }

  def getPrecisionRecall(dataframe: DataFrame) = {

    val groundPositive = dataframe.filter(col("label") === 1.0).count()

    val predictSummary = getCutPointSummary(dataframe)
    val predictLabelSummary = getCutPointSummary(dataframe.filter(col("label") === 1.0))

    predictSummary.map(predictedPositive => {
      val truePositive = predictLabelSummary(predictedPositive._1)
      val precision = truePositive / predictedPositive._2
      val recall = truePositive / groundPositive
      (predictedPositive._1, precision, recall)
    }
    ).toArray.sortBy(x => x._1)
  }

  def getCutPointSummary(dataframe: DataFrame): Map[Float, Float] = {

    dataframe.select("score").rdd
      .map(x => x.getFloat(0))
      .flatMap(x => {
        (0.02 to 0.98 by 0.02).map(cutPoint => if (x > cutPoint) (cutPoint, 1.0) else (cutPoint, 0.0))
      }).reduceByKey(_ + _).collect().map(x => (x._1.toFloat, x._2.toFloat)).toMap

  }


  def getAbsRank(df: DataFrame, K: Int = 30): DataFrame = {

    val ranked = if (!df.columns.contains("rank")) {
      val w2 = Window.partitionBy("userIdIndex").orderBy(desc("score"))
      df.withColumn("rank", rank.over(w2)).where(col("rank") <= K)
    } else {
      df.where(col("rank") <= K)
    }

    ranked.registerTempTable("temp")

    val labeled = df.sqlContext.sql("select userIdIndex, avg(rank) from temp where label = 1.0 group by userIdIndex")

    val dict = labeled.select("userIdIndex").rdd.map(row => row.getDouble(0)).collect().toSet

    val filterUdf = udf((userIdIndex: Double) => !dict.contains(userIdIndex))
    val noLabel = ranked.filter(filterUdf(col("userIdIndex"))).select("userIdIndex")
      .distinct().withColumn("avg(rank)", lit(K + 1))

    val rankDF = labeled.union(noLabel)

    val roundUDF = udf((v: Double) => v.toInt)
    rankDF.withColumn("roundRank", roundUDF(col("avg(rank)"))).groupBy("roundRank")
      .count().orderBy(col("roundRank"))
  }


  def getHitRatioNDCG(dfin: DataFrame, K: Int = 30): (Double, Double) = {

    val w2 = Window.partitionBy("userIdIndex").orderBy(desc("score"))
    val ranked = dfin.withColumn("rank", rank.over(w2)).where(col("rank") <= K)

    val w1 = Window.partitionBy("userIdIndex").orderBy(desc("label"))
    val selected = ranked.withColumn("rn", row_number.over(w1)).where(col("rn") === 1).drop("rn")

    val ndcgUdf = udf((rank: Int, label: Int) => if (label == 1) math.log(2) / math.log(rank + 1) else 0)

    val df = selected.withColumn("hit", when(col("label").isNull or col("label") === 0, 0).otherwise(1))
      .withColumn("ndcg", when(col("label").isNotNull, ndcgUdf(col("rank"), col("label"))).otherwise(0))

    df.filter("label = 0").count()

    val resultDF = df.groupBy("userIdIndex")
      .agg(sum("hit"), sum("ndcg"))
      .agg(avg(col("sum(hit)")), avg("sum(ndcg)"))

    val r = resultDF.rdd.take(1).map(row => (row.getDouble(0), row.getDouble(1)))

    r(0)
  }

  val score2bucket = {
    val func = (arg: Float) => (arg * 10).toInt
    udf(func)
  }

  def bucketize(df: DataFrame) = {

    def num2percent(total: Long) = {
      val func = (arg: Long) => arg.toFloat / total
      udf(func)
    }

    val groundPositive = df.filter(col("label") === 1.0).count()

    df.filter(col("label") === 1.0)
      .withColumn("bucket", score2bucket(col("score")))
      .groupBy("bucket").count()
      .withColumn("percent", num2percent(groundPositive)(col("count")))
  }

  def toDecimal(n: Int) = {
    (arg: Double) => BigDecimal(arg).setScale(n, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

}
