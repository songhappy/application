package com.intel.analytics.zoo.apps.tutorial

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, Initializable, TensorModule}
import com.intel.analytics.bigdl.nn.quantized.Quantizable
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils._

import scala.concurrent.Future
import scala.reflect.ClassTag
import breeze.linalg.Axis._1
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.FunSuite

import scala.util.Random


class understandData extends FunSuite {

  test("understanding tensor") {
    val tensor: Tensor[Float] = Tensor(2, 2, 3).apply1(e => Random.nextFloat())
    println(tensor.storage())
    println(tensor.storageOffset())
    println(tensor.size().mkString("|"))
    println(tensor.stride().mkString("|"))

    println("--------------------")
    println(tensor)
    println("-----------------")
    println(tensor.select(1, 1))
    println("-----------------")
    println(tensor.select(2, 1).transpose(1, 1))
    println("-----------------")
    println(tensor.transpose(1, 2))
  }

  test("dimension and input and window") {

    val kernelW = 5
    val strideW = 2
    val outputFrameStride = (kernelW - 1) / strideW + 1
    val inputFrameStride = outputFrameStride * strideW
    //val inputoffset = j * strideW * input.size(dimFeat)
    //val outputoffset = output.storageOffset() + j * output.size(dimFeat)
    val tensor = Tensor(100, 10)
    for (i <- 0 to 999) {
      val row = i / 10 +1
      val col = i % 10 +1
      println(row, col)
      tensor.setValue(row, col, i)
    }
    println(tensor)

    val window = Tensor()


    val inputoffset = strideW * 10
    // val outputoffset = j * 8
    //  println(tensor)
    //  println(window.set(tensor.storage(), storageOffset = 1, Array(1, 50), strides = Array(1, 1)))
    //  println(window.set(tensor.storage(), storageOffset = 21, Array(1, 50), strides = Array(1, 1)))

    val weight2 = Tensor(8, 50)
    for (i <- 0 to 399) {
      val row = i / 50 + 1
      val col = i % 50 + 1
      weight2.setValue(row, col, i)
    }

    val windowWeight2 = Tensor()
    // println(weight2)
    //  println(window.set(weight2.storage(), storageOffset = 1, Array(8, 50), strides = Array(50, 1)))

    val d1l = 48
    val d2l = 8
    val d3l = 50
    val weight = Tensor(d1l, d2l, d3l)
    val page = d2l * d3l
    for (i <- 0 to d1l * d2l * d3l - 1) {
      val d1 = i / page + 1
      val d2 = (i % page) / (d3l) + 1
      val d3 = (i % page) % d3l + 1
      weight.setValue(d1, d2, d3, i)
    }
    print("weight")
    println(weight)

    val windowweight = Tensor()
    println(windowweight.set(weight.storage(), storageOffset = 1, Array(8, 50), strides = Array(50, 1)))
    println(windowweight.set(weight.storage(), storageOffset = 401, Array(8, 50), strides = Array(50, 1)))

    //println(window.set(weight.storage(), storageOffset = 1, Array(8, 50, 1), strides = Array(page +60, 1, 1)))
    //println(window.set(weight.storage(), storageOffset = 21, Array(8, 50, 1), strides = Array(page + 60, 1, 1)))

    //    println(tensor)
    //    val window = Tensor()
    //    println(window.set(tensor.storage(), storageOffset = 21, Array(16, 50), strides = Array(60, 1)))
    //   println(tensor.narrow(2, 2, 2))
  }

  test("Tensor1") {
    val t1 = Tensor(1, 2, 3)
    println(t1)

    val t2: Array[Float] = Tensor(3).toArray()
    // val values: Float = t2.value()

    println(t2)
    // println(values)

  }

  test("Tensor") {

    val tensor2: Tensor[Float] = Tensor(T(8.0.toFloat, 2.0.toFloat))
    val a: Tensor[Float] = Tensor(T(
      T(1f, 2f, 3f),
      T(4f, 5f, 6f)))
    val t1: Table = T(2, 3, 4)

    println(a)
    println(tensor2)
    println(t1)
    println(T(Tensor(2, 2).fill(1), Tensor(2, 2).fill(2)))

    val x: Storage[Float] = a.storage()
    println(a.storage())

  }

  test("Table") {
    val tt = T(1f, 2f, 3f)
    println(tt.get(1))
  }

  test("Sample") {
    val image = Tensor(3, 32, 32).rand
    val label = 1f
    val sample = Sample(image, label)
    val tensor2: Tensor[Float] = Tensor(T(1.0.toFloat, 2.0.toFloat))

    val sample2 = Sample(tensor2, label)

    tensor2.toTable.toSeq

    println(sample2.getData().mkString(","))

  }

  test("MiniBatch") {
    import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
    import com.intel.analytics.bigdl.numeric.NumericFloat
    import com.intel.analytics.bigdl.tensor.Tensor

    val samples = Array.tabulate(10)(i => Sample(Tensor(1, 3, 3).fill(i), i + 10f))
    val miniBatch = MiniBatch(1, 1).set(samples)
    println(miniBatch.getInput())
    println(miniBatch.getTarget())

  }


  test("model") {
    val model = Sequential().add(Linear(10, 5)).add(Sigmoid()).add(SoftMax())

  }
  test("reshape") {
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.tensor.Tensor
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

    val reshape = Reshape(Array(4, 2, 3))
    val input = Tensor(2, 3, 4).rand()
    val output = reshape.forward(input)
    println(input.size().toList)
    print(output.size().toList)
  }

  test("select") {
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.tensor.Tensor
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    import com.intel.analytics.bigdl.utils.T

    val layer = Select(2, 1)
    val output1 = layer.forward(Tensor(T(
      T(1.0f, 2.0f, 3.0f),
      T(4.0f, 5.0f, 6.0f),
      T(7.0f, 8.0f, 9.0f)
    )))

    val output2 = layer.backward(Tensor(T(
      T(1.0f, 2.0f, 3.0f),
      T(4.0f, 5.0f, 6.0f),
      T(7.0f, 8.0f, 9.0f)
    )), Tensor(T(1.0f, 4f, 7f)))

    println(output1)
    println(output2)
  }

  test("layer functions") {
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.tensor.Tensor
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

    val layer = Sigmoid()
    val input = Tensor(2, 3)

    var i = 0
    val mid = input.apply1(_ => {
      i += 1;
      i
    })

    val output1 = layer.forward(mid)
    val output2 = layer.backward(mid, output1)

    println(mid)
    println(output1)

    println(output2)
  }

  test("softmax functions") {
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.tensor.Tensor
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

    val layer = SoftMax[Float]
    val input = Tensor(2, 3)

    var i = 0
    val mid = input.apply1(_ => {
      i += 1;
      i
    })

    val output1 = layer.forward(mid)
    val output2 = layer.backward(mid, output1)

    println(mid)
    println(output1)

    println(output2)
  }
  test("lookuptable") {
    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    import com.intel.analytics.bigdl.tensor._

    val layer = LookupTable(nIndex = 9, nOutput = 4, paddingValue = 2, maxNorm = 0.1, normType = 2.0, shouldScaleGradByFreq = true)

    val input1 = Tensor(
      T(3f, 4f, 5f, 8f, 7f)
    )

    val input = Tensor(Storage(Array(5.0f, 2.0f, 6.0f, 9.0f, 4.0f, 8.0f)), 1, Array(5), Array(1))
    val output = layer.forward(input1)
    //val gradInput = layer.backward(input, output)

    println(input)
    //    println(layer)
    //    println(input1)
    //    println(input1.size().toList)
    //    println(output)
    //    println(output.size().toList)toList
    // println(gradInput)
  }


  test("sparse tensor") {
  }

}
