package com.intel.analytics.zoo.apps.recommendation

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame
import com.intel.analytics.zoo.apps.recommendation.Utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._

object Evaluation {

  def evaluate(evaluateDF: DataFrame) = {

    val binaryEva = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
    val out = binaryEva.evaluate(evaluateDF)
    println("AUROC: " + toDecimal(3)(out))

    val multiEva1 = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val out1 = multiEva1.evaluate(evaluateDF)
    println("accuracy: " + toDecimal(3)(out1))


    val multiEva2 = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision")
    val out2 = multiEva2.evaluate(evaluateDF)
    println("precision: " + toDecimal(3)(out2))

    val multiEva3 = new MulticlassClassificationEvaluator().setMetricName("weightedRecall")
    val out3 = multiEva3.evaluate(evaluateDF)
    println("recall: " + toDecimal(3)(out3))

    Seq(out, out1, out2, out3).map(x => toDecimal(3)(x))
  }

  def evaluate2(evaluateDF: DataFrame) = {
    val truePositive = evaluateDF.filter(col("prediction") === 1.0 && col("label") === 1.0).count()
    val falsePositive = evaluateDF.filter(col("prediction") === 1.0 && col("label") === 0.0).count()
    val trueNegative = evaluateDF.filter(col("prediction") === 0.0 && col("label") === 0.0).count()
    val falseNegative = evaluateDF.filter(col("prediction") === 0.0 && col("label") === 1.0).count()
    val accuracy = (truePositive.toDouble + trueNegative.toDouble) / (trueNegative.toDouble + truePositive.toDouble + falseNegative.toDouble + falsePositive.toDouble)
    val precision = truePositive.toDouble / (truePositive.toDouble + falsePositive.toDouble)
    val recall = truePositive.toDouble / (truePositive.toDouble + falseNegative.toDouble)

    println("truePositive: " + truePositive)
    println("falsePositive: " + falsePositive)
    println("trueNegative: " + trueNegative)
    println("falseNegative: " + falseNegative)
    println("accuracy: " + accuracy)
    println("precision: " + precision)
    println("recall: " + recall)

    val evaluation = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
    val evaluatLabels = evaluateDF
      .withColumn("label", col("label").cast("double"))
      .withColumn("prediction", col("prediction").cast("double"))
    val modelAUROC = evaluation.setMetricName("areaUnderROC").evaluate(evaluatLabels)
    val modelAUPR = evaluation.setMetricName("areaUnderPR").evaluate(evaluatLabels)
    println("modelAUROC: " + modelAUROC)
    println("modelAUPR: " + modelAUPR)
    Seq(accuracy, precision, recall, modelAUROC, modelAUPR).map(x => toDecimal(3)(x))

  }

}
