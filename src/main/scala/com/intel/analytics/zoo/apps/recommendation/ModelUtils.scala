package com.intel.analytics.zoo.apps.recommendation

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.apps.recommendation.Utils._
import org.apache.spark.ml.DLModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

case class ModelParam(userEmbed: Int = 20,
                      itemEmbed: Int = 20,
                      mfEmbed: Int = 20,
                      path:String = "",
                      midLayers: Array[Int] = Array(40, 20, 10),
                      labels: Int = 2) {
  override def toString: String = {
    "userEmbed =" + userEmbed + "\n" +
      " itemEmbed = " + itemEmbed + "\n" +
      " mfEmbed = " + mfEmbed + "\n" +
      " midLayer = " + midLayers.mkString("|") + "\n" +
      " labels = " + labels
  }
}

case class featuresScore(feature1: Int, feature2: Int, score: Double)

class ModelUtils(modelParam: ModelParam) {

  def this() = {
    this(ModelParam())
  }

  def mlp(userCount: Int, itemCount: Int): Graph[Float] = {

    println(modelParam)

    val input = Identity().inputs()
    val select1: ModuleNode[Float] = Select(2, 1).inputs(input)
    val select2: ModuleNode[Float] = Select(2, 2).inputs(input)

    val userTable = LookupTable(userCount, modelParam.userEmbed)
    val itemTable = LookupTable(itemCount, modelParam.itemEmbed)
    userTable.setWeightsBias(Array(Tensor[Float](userCount, modelParam.userEmbed).randn(0, 0.1)))
    itemTable.setWeightsBias(Array(Tensor[Float](itemCount, modelParam.itemEmbed).randn(0, 0.1)))

    val userTableInput = userTable.inputs(select1)
    val itemTableInput = itemTable.inputs(select2)

    val embeddedLayer = JoinTable(2, 0).inputs(userTableInput, itemTableInput)

    val linear1: ModuleNode[Float] = Linear(modelParam.itemEmbed + modelParam.userEmbed,
      modelParam.midLayers(0)).inputs(embeddedLayer)

    val midLayer = buildMlpModuleNode(linear1, 1, modelParam.midLayers)

    val reluLast = ReLU().inputs(midLayer)
    val last: ModuleNode[Float] = Linear(modelParam.midLayers.last, modelParam.labels).inputs(reluLast)

    val output = if (modelParam.labels >= 2) LogSoftMax().inputs(last) else Sigmoid().inputs(last)

    Graph(input, output)
  }

  private def buildMlpModuleNode(linear: ModuleNode[Float], midLayerIndex: Int, midLayers: Array[Int]): ModuleNode[Float] = {

    if (midLayerIndex >= midLayers.length) {
      linear
    } else {
      val relu = ReLU().inputs(linear)
      val l = Linear(midLayers(midLayerIndex - 1), midLayers(midLayerIndex)).inputs(relu)
      buildMlpModuleNode(l, midLayerIndex + 1, midLayers)
    }

  }

  def mlpSeq(userCount:Int, itemCount:Int): Sequential[Float] = {
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
   //   .add(Sequential().add(Select(2,3)).add(View(1, 1)))

    val model = Sequential()
    model.add(embedded_layer)

    model.add(Linear(userOutput + itemOutput, linear1)).add(ReLU())
    model.add(Linear(linear1, modelParam.labels))
    model.add(LogSoftMax())

    model
  }


  def mlp2 = {

    println(modelParam)

    val input = Identity().inputs()

    // val linear1: ModuleNode[Float] = Linear(modelParam.itemEmbed + modelParam.userEmbed,
    val linear1: ModuleNode[Float] = Linear(100,
      40).inputs(input)

    val relu1 = ReLU().inputs(linear1)
    val linear2 = Linear(40, 20).inputs(relu1)

    val relu2 = ReLU().inputs(linear2)
    val linear3 = Linear(20, 10).inputs(relu2)

    val reluLast = ReLU().inputs(linear3)
    val last: ModuleNode[Float] = Linear(10, 2).inputs(reluLast)

    val output = if (modelParam.labels >= 2) LogSoftMax().inputs(last) else Sigmoid().inputs(last)

    Graph(input, output)
  }

  def mlp3 = {
    val model = Sequential()
    model.add(Linear(100, 40))
    model.add(ReLU())
    model.add(Linear(40, 20))
    model.add(ReLU())
    model.add(Linear(20, 10))
    model.add(ReLU())
    model.add(Linear(10, 2))
    model.add(ReLU())
    model.add(LogSoftMax())
    model
  }

  def ncf(userCount: Int, itemCount: Int) = {

    val mfUserTable = LookupTable(userCount, modelParam.mfEmbed)
    val mfItemTable = LookupTable(itemCount, modelParam.mfEmbed)
    val mlpUserTable = LookupTable(userCount, modelParam.userEmbed)
    val mlpItemTable = LookupTable(itemCount, modelParam.itemEmbed)

    val mfEmbeddedLayer = ConcatTable().add(Sequential().add(Select(2, 1)).add(mfUserTable))
      .add(Sequential().add(Select(2, 2)).add(mfItemTable))

    val mlpEmbeddedLayer = Concat(2).add(Sequential().add(Select(2, 1)).add(mlpUserTable))
      .add(Sequential().add(Select(2, 2)).add(mlpItemTable))

    val mfModel = Sequential()
    mfModel.add(mfEmbeddedLayer).add(CMulTable())

    val mlpModel = Sequential()
    mlpModel.add(mlpEmbeddedLayer)

    val linear1 = Linear(modelParam.itemEmbed + modelParam.userEmbed, modelParam.midLayers(0))
    mlpModel.add(linear1).add(ReLU())


    for (i <- 1 to modelParam.midLayers.length - 1) {
      mlpModel.add(Linear(modelParam.midLayers(i - 1), modelParam.midLayers(i))).add(ReLU())
    }

    val concatedModel = Concat(2).add(mfModel).add(mlpModel)

    val model = Sequential()
    model.add(concatedModel).add(Linear(modelParam.mfEmbed + modelParam.midLayers.last, modelParam.labels))

    if (modelParam.labels >= 2) model.add(LogSoftMax()) else model.add(Sigmoid())
    model
  }

  /**
    * Returns top `numItems`  recommended for each user, for usersForRec users.
    *
    * @param usersForRec a dataframe of users who want to be recommended
    * @param itemsForRec a dataframe of items
    * @param indexed     original data input
    * @param dlModel     a trained deep learning model
    * @param numItems    max number of recommendations for each item
    * @return a DataFrame of (itemIdIndex: Int, recommendations), where recommendations are
    *         stored as an array of (userIdIndex: Int, rating: Float) Rows.
    */
  def recommendForUsers(usersForRec: DataFrame,
                        itemsForRec: DataFrame,
                        indexed: DataFrame,
                        dlModel: DLModel[Float],
                        numItems: Int): DataFrame = {
    val predictions = rankForNegative(usersForRec, itemsForRec, indexed, dlModel)

    recommend(predictions, numItems, true)
  }

  /**
    * Returns top `numUsers` users recommended for each item, for itemsForRec items.
    *
    * @param usersForRec a dataframe of users who want to be recommended
    * @param itemsForRec a dataframe of items
    * @param indexed     original data input
    * @param dlModel     a trained deep learning model
    * @param numUsers    max number of recommendations for each item
    * @return a DataFrame of (itemIdIndex: Int, recommendations), where recommendations are
    *         stored as an array of (userIdIndex: Int, rating: Float) Rows.
    */
  def recommendForItems(usersForRec: DataFrame,
                        itemsForRec: DataFrame,
                        indexed: DataFrame,
                        dlModel: DLModel[Float],
                        numUsers: Int): DataFrame = {
    val predictions = rankForNegative(usersForRec, itemsForRec, indexed, dlModel)

    recommend(predictions, numUsers, false)
  }

  private def recommend(predictions: DataFrame, numItems: Int, forUser: Boolean = true): DataFrame = {
    import predictions.sqlContext.implicits._
    predictions.rdd.map(row =>
      if (forUser) {
        featuresScore(row.getAs[DenseVector](1)(0).toInt,
          row.getAs[DenseVector](1)(1).toInt, row.getDouble(2))
      } else {
        featuresScore(row.getAs[DenseVector](1)(1).toInt,
          row.getAs[DenseVector](1)(0).toInt, row.getDouble(2))
      })
      .groupBy(x => x.feature1)
      .map(x => (x._1, x._2.toList.sortBy(x => x.score).reverse.map(x => (x.feature2, x.score)).take(numItems)))
      .toDF("id", "recommendations")
  }

  def rankForNegative(usersForRec: DataFrame, itemsForRec: DataFrame, indexed: DataFrame, dlModel: DLModel[Float]) = {
    val all = usersForRec.select("userIdIndex").crossJoin(itemsForRec.select("itemIdIndex"))
      .except(indexed.select("userIdIndex", "itemIdIndex"))
    dlModel.transform(df2LP(all.withColumn("label", lit(0.0d))))
  }
}
