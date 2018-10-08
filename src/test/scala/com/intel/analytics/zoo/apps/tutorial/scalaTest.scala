package com.intel.analytics.zoo.apps.tutorial

import org.scalatest.FunSuite

class scalaTest  extends FunSuite {

  test("array") {
    val arr = Array(1,2,3,4,2,3,6)
    val x = arr.toSet.size
    val y = arr.map(x => (x, 1))
      .groupBy(x=> x._1)
      .map(x=> (x._1, x._2.size))
  }

}
