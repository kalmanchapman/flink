/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.optimization

import com.github.fommil.netlib.BLAS.{getInstance => blas}

import breeze.{numerics => BreezeNumerics}
import org.apache.flink.api.common._
import org.apache.flink.api.scala._
import org.apache.flink.api.scala.utils._
import org.apache.flink.ml._
import org.apache.flink.ml.common.Parameter
//import org.apache.flink.ml.math.{BLAS, DenseVector}
import org.apache.flink.ml.optimization.Embedder._
import org.apache.flink.util.XORShiftRandom

import scala.reflect.ClassTag

/**
  * some description of a huffman binary tree and why we use it
  */

object HuffmanBinaryTree extends Serializable {
  import collection.mutable

  abstract class Tree[+A]

  private type WeightedNode = (Tree[Any], Int)

  implicit def orderByInverseWeight[A <: WeightedNode]: Ordering[A] =
    Ordering.by {
      case (_, weight) => -1 * weight
    }

  private case class Leaf[A](value: A) extends Tree[A]

  private case class Branch[A](left: Tree[A], right: Tree[A]) extends Tree[A]

  // recursively build the binary tree needed to Huffman encode the text
  // not immutable :(
  private def merge(xs: mutable.PriorityQueue[WeightedNode])
  : mutable.PriorityQueue[WeightedNode] = {
    if (xs.length == 1) xs
    else {
      val l = xs.dequeue
      val r = xs.dequeue
      val merged = (Branch(l._1, r._1), l._2 + r._2)
      merge(xs += merged)
    }
  }

  //convenience for building and extracting from priority queue
  private def merge(xs: Iterable[WeightedNode]): Iterable[WeightedNode] = {
    //form priority queue, build tree, return a list (should have only 1 member)
    merge(new mutable.PriorityQueue[WeightedNode] ++= xs).toList
  }

  // recursively search the branches of the tree for the required character
  private def contains(tree: Tree[Any], value: Any): Boolean = tree match {
    case Leaf(c) => if (c == value) true else false
    case Branch(l, r) => contains(l, value) || contains(r, value)
  }

  // recursively build the huffman encoding of a token value
  def encode(tree: Tree[Any], value: Any): Vector[Int] = {
    def turn(tree: Tree[Any], value: Any,
             code: Vector[Int]): Vector[Int] = tree match {
      case Leaf(_) => code
      case Branch(l, r) =>
        (contains(l, value)) match {
          case true => turn(l, value, code :+ 0)
          case _ => turn(r, value, code :+ 1)
        }
    }
    turn(tree, value, Vector.empty[Int])
  }

  //takes a code and decomposes into visited node identities
  def path(code: Vector[Int]) : Vector[String] = {
    def form(code: Vector[Int], path: Vector[String]): Vector[String] = code match {
      case _ +: IndexedSeq() => path.reverse
      case head +: tail => path match {
        case _ +: IndexedSeq() => form(tail, head.toString +: path)
        case _ => form(tail, (path.head + head) +: path)
      }
    }
    form(code, Vector("root"))
  }

  def tree(weightedLexicon: Iterable[(Any, Int)]): Tree[Any] =
    merge(weightedLexicon.map(x => (Leaf(x._1), x._2))).head._1
}

/**
  * explanations of the supporting classes & traits?
  */

trait WeightMatrix

trait TrainingSet

case class Context[T](target: T, context: Iterable[T])

case class HSMTargetValue(index: Long, code: Vector[Int], path: Vector[String])

case class HSMStepValue(innerIndex: Vector[Long], code: Vector[Int], codeDepth: Int)

case class HSMTrainingSet(leafSet: Vector[Long], innerSet: HSMStepValue) extends TrainingSet

case class HSMWeightMatrix[T](leafMap: Map[T, HSMTargetValue],
                              innerMap: Map[String, Long],
                              leafVectors: Array[Double],
                              innerVectors: Array[Double]) extends WeightMatrix {
  def ++ (that: HSMWeightMatrix[T]): HSMWeightMatrix[T] = {
    HSMWeightMatrix(this.leafMap ++ that.leafMap,
      this.innerMap ++ that.innerMap,
      that.leafVectors,
      that.innerVectors)
  }

  def ++ (vectors: (Array[Double], Array[Double])): HSMWeightMatrix[T] = {
    HSMWeightMatrix(this.leafMap, this.innerMap, vectors._1, vectors._2)
  }
}

/**
  * Implements Word2Vec; a word-embedding algoritm first described
  * by Tomáš Mikolov et al in http://arxiv.org/pdf/1301.3781.pdf
  *
  *
  *
  * Hierarchical SoftMax is an approach to the softmax classification solver
  * that utilizes a distributed, sequential representation of class output probabilities
  * via a huffman encoding of known classes and a training process that 'learns'
  * the output classes by traversing the inner probabilities of the encoding
  *
  * More on hierarchical softmax: (http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
  * More on softmax: (https://en.wikipedia.org/wiki/Softmax_function)
  */

object Embedder {
  val MIN_LEARNING_RATE = 0.0001

  case object Iterations extends Parameter[Int] {
    val defaultValue = Some(10)
  }

  case object TargetCount extends Parameter[Int] {
    val defaultValue = Some(5)
  }

  case object VectorSize extends Parameter[Int] {
    val defaultValue = Some(100)
  }

  case object LearningRate extends Parameter[Double] {
    val defaultValue = Some(0.015)
  }

  case object Seed extends Parameter[Long] {
    val defaultValue = Some(scala.util.Random.nextLong)
  }

  case object BatchSize extends Parameter[Int] {
    val defaultValue = Some(1000)
  }

}

abstract class Embedder[A, B] extends Solver[A, B] {
  import Embedder._
  def setIterations(iterations: Int): this.type = {
    parameters.add(Iterations, iterations)
    this
  }

  def setTargetCount(targetCount: Int): this.type = {
    parameters.add(TargetCount, targetCount)
    this
  }

  def setVectorSize(vectorSize: Int): this.type = {
    parameters.add(VectorSize, vectorSize)
    this
  }

  def setLearningRate(learningRate: Double): this.type = {
    parameters.add(LearningRate, learningRate)
    this
  }

  def setBatchSize(batchSize: Int): this.type = {
    parameters.add(BatchSize, batchSize)
    this
  }

  def setSeed(seed: Long): this.type = {
    parameters.add(Seed, seed)
    this
  }
}

class ContextEmbedder[T: ClassTag: typeinfo.TypeInformation]
  extends Embedder[Context[T], HSMWeightMatrix[T]] {

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6

  val numberOfIterations: Int = parameters(Iterations)
  val minTargetCount: Int = parameters(TargetCount)
  val vectorSize: Int = parameters(VectorSize)
  val learningRate: Double = parameters(LearningRate)
  val batchSize: Int = parameters(BatchSize)
  val seed: Long = parameters(Seed)

  def optimize(data: DataSet[Context[T]],
               initialWeights: Option[DataSet[HSMWeightMatrix[T]]]): DataSet[HSMWeightMatrix[T]] = {

    val weights: DataSet[HSMWeightMatrix[T]] = createInitialWeightsDS(initialWeights, data)

    val vocab = weights.map(x => x.leafMap.keySet)

    val vocabSize = vocab.flatMap(x => x).count().toInt

    val preparedData = data
      .filterWithBcVariable(vocab){
        (context, vocab) =>
          vocab.contains(context.target)
      }
      .mapWithBcVariable(weights){
        (context, weights) =>
          val target = weights.leafMap.get(context.target).get
          val path = target.path
          val code = target.code
          val codeDepth = code.size
          val contextIndices = context.context.flatMap(x =>
            weights.leafMap.get(x) match {
              case Some(value) => Some(value.index)
              case None => None
            }).to[Vector]
          val pathIndices = path.flatMap(x =>
            weights.innerMap.get(x) match {
              case Some(index) => Some(index)
              case None => None
            })
          HSMTrainingSet(contextIndices, HSMStepValue(pathIndices, code, codeDepth))
      }.filter(_.leafSet.nonEmpty)

    weights.iterate(numberOfIterations) {
      w => trainIteration(preparedData, w, vocabSize, learningRate)
    }
  }

  def createInitialWeightsDS(initialWeights: Option[DataSet[HSMWeightMatrix[T]]],
                              data: DataSet[Context[T]])
  : DataSet[HSMWeightMatrix[T]] = initialWeights match {
    case Some(weightMatrix) => weightMatrix
    case None => formHSMWeightMatrix(data)
  }

  private def formHSMWeightMatrix(data: DataSet[Context[T]])
  : DataSet[HSMWeightMatrix[T]] = {
    val env = data.getExecutionEnvironment

    val targets = data
      .map(x => (x.target, 1))
      .groupBy(0).sum(1)
      .filter(_._2 >= minTargetCount)

    val softMaxTree = env.fromElements(HuffmanBinaryTree.tree(targets.collect()))

    val leafValues = targets
      .mapWithBcVariable(softMaxTree) {
        (target, softMaxTree) => {
          val code = HuffmanBinaryTree.encode(softMaxTree, target._1)
          val path = HuffmanBinaryTree.path(code)
          target._1 -> (code, path)
        }
      }.zipWithIndex

    val leafCount = leafValues.count()

    val initRandom = new XORShiftRandom(seed)

    val innerMap = leafValues
      .flatMap(x => x._2._2._2)
      .distinct()
      .zipWithIndex
      .map(x => Map(x._2 -> x._1))
      .reduce(_ ++ _)
      .map(m => HSMWeightMatrix(Map.empty[T, HSMTargetValue], m, Array.empty, Array.empty))

    val leafMap = leafValues
      .map(x => Map(x._2._1 -> HSMTargetValue(x._1, x._2._2._1, x._2._2._2)))
      .reduce(_ ++ _)
      .map(m => HSMWeightMatrix(m,Map.empty,Array.empty, Array.empty))

    innerMap.union(leafMap).reduce(_ ++ _)
      .map(x => {
        val lVectors =
          Array.fill[Double](leafCount.toInt * vectorSize)(
            (initRandom.nextDouble() - 0.5f) / vectorSize)
        val iVectors = new Array[Double](leafCount.toInt * vectorSize)
        x.copy(leafVectors = lVectors, innerVectors = iVectors)
      })
  }

  private def trainIteration(data: DataSet[HSMTrainingSet],
                       weights: DataSet[HSMWeightMatrix[T]],
                       vocabSize: Int,
                       learningRate: Double)
  : DataSet[HSMWeightMatrix[T]] = {
    val learnedWeights = data
      .zipWithIndex
      .map(x => x._1 % batchSize -> Seq(x._2))
      .groupBy(0)
      .reduce((a,b) => (a._1, a._2 ++ b._2))
      .map(_._2)
      .mapWithBcVariable(weights)(train)

    val learnedLeafWeights = aggregateWeights(
      learnedWeights.map(_._1).flatMap(x => x),
      vocabSize)

    val learnedInnerWeights = aggregateWeights(
      learnedWeights.map(_._2).flatMap(x => x),
      vocabSize)

    learnedLeafWeights
      .crossWithTiny(learnedInnerWeights)
      .crossWithTiny(weights)
      .map(x => x._2 ++ x._1)
  }

  private def aggregateWeights(weights: DataSet[(Int, Array[Double])], vocabSize: Int)
  : DataSet[Array[Double]] = weights
    .groupBy(0)
    .reduce(sumWeights(_,_))
    .map(x => {
      val globalVector = new Array[Double](vocabSize * vectorSize)
      x._2.copyToArray(globalVector, x._1 * vectorSize)
      Seq(x._1) -> globalVector
    })
    .reduce(mergeWeights(_,_))
    .map(_._2)

  private def sumWeights[V <: (Int, Array[Double])](vecA: V, vecB: V)  = {
    val targetVector = vecB._2
    blas.daxpy(vectorSize, 1.0d, vecA._2, 1, targetVector, 1)
    (vecB._1, targetVector)
  }

  def mergeWeights[V <: (Seq[Int], Array[Double])](vecA: V, vecB: V) = {
    val sinkVector = vecA._2.clone()
    vecB._1.foreach(
      index => {
        val vecInd = index * vectorSize
        Array.copy(vecB._2, vecInd, sinkVector, vecInd, vectorSize)
    })
    vecA._1 ++ vecB._1 -> sinkVector
  }

  private def train(
    context: Seq[HSMTrainingSet],
    weights: (Array[Double], Array[Double]),
    alpha: Option[Double])
  : (Seq[(Int, Array[Double])], Seq[(Int, Array[Double])]) = {
    val expTable = createExpTable()
    val vocabSize = weights._1.length / vectorSize
    val leafModify = new Array[Int](vocabSize)
    val innerModify = new Array[Int](vocabSize)
    val initialAlpha = alpha.getOrElse(learningRate)
    val count = context.size
    val model = context.foldLeft((weights._1, weights._2, initialAlpha, 0, count)) {
      case ((leafWeights, innerWeights, a, trainSetPos, trainSetSize), trainingSet) =>
        val decayedAlpha = (a * (1 - trainSetPos / trainSetSize)).max(MIN_LEARNING_RATE)
        val contextSize = trainingSet.leafSet.size
        var contextPos = 0
        while (contextPos < contextSize) {
          val leafIndex = trainingSet.leafSet(contextPos).toInt
          val leafWeightIndex = leafIndex * vectorSize
          val hiddenVector = new Array[Double](vectorSize)
          var codePos = 0
          while (codePos < trainingSet.innerSet.codeDepth) {
            val innerIndex = trainingSet.innerSet.innerIndex(codePos).toInt
            val innerWeightIndex = innerIndex * vectorSize
            var forwardPass =
              blas.ddot(vectorSize, leafWeights, leafWeightIndex, 1,
                innerWeights, innerWeightIndex, 1)
            if (forwardPass > -MAX_EXP && forwardPass < MAX_EXP) {
              val expIndex = ((forwardPass + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
              forwardPass = expTable(expIndex)
              val gradient =
                (1 - trainingSet.innerSet.code(codePos) - forwardPass) * decayedAlpha
              blas.daxpy(vectorSize, gradient, innerWeights, innerWeightIndex, 1,
                hiddenVector, 0, 1)
              blas.daxpy(vectorSize, gradient, leafWeights, leafWeightIndex, 1,
                innerWeights, innerWeightIndex, 1)
              innerModify.update(innerIndex, 1)
            }
            codePos += 1
          }
          blas.daxpy(vectorSize, 1.0f, hiddenVector, 0, 1,
            leafWeights, leafWeightIndex, 1)
          leafModify.update(leafIndex, 1)
          contextPos += 1
        }
        (leafWeights, innerWeights, decayedAlpha, trainSetPos + 1, trainSetSize)
    }
    val leafW = model._1
    val innerW = model._2
    val sparseLeaf = Iterator.tabulate(vocabSize) {
      index =>
        if (leafModify(index) > 0) {
          Some(index, leafW.slice(index * vectorSize, (index + 1) * vectorSize))
        } else {
          None
        }
    }.flatten.toSeq
    val sparseInner = Iterator.tabulate(vocabSize) {
      index =>
        if (innerModify(index) > 0) {
          Some(index, innerW.slice(index * vectorSize, (index + 1) * vectorSize))
        } else {
          None
        }
    }.flatten.toSeq
    sparseLeaf -> sparseInner
  }

  private def train(
  context: Seq[HSMTrainingSet],
  weights: HSMWeightMatrix[T])
  : (Seq[(Int, Array[Double])], Seq[(Int, Array[Double])]) =
    train(context, weights.leafVectors -> weights.innerVectors, None)

  private def createExpTable(): Array[Double] = {
    val expTable = new Array[Double](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = scala.math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = tmp / (tmp + 1.0)
      i += 1
    }
    expTable
  }
}
