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

import breeze.{numerics => BreezeNumerics}
import org.apache.flink.api.common._
import org.apache.flink.api.scala._
import org.apache.flink.api.scala.utils._
import org.apache.flink.ml._
import org.apache.flink.ml.common.Parameter
import org.apache.flink.ml.math.{BLAS, DenseVector}
import org.apache.flink.ml.optimization.Embedder._

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

case class HSMTargetValue(vector: DenseVector, count: Int, code: Vector[Int], path: Vector[String])

case class HSMStepValue(key: String, target: Int, vector: DenseVector)

case class HSMTrainingSet[T](leaf: Seq[T],
                             inner: Seq[String]) extends TrainingSet

case class LocalWeightMatrix[T](leafVectors: Map[T, DenseVector],
                                innerVectors: Map[String, DenseVector]) extends WeightMatrix {
  def ++ (that: LocalWeightMatrix[T]): LocalWeightMatrix[T] = {
    LocalWeightMatrix(this.leafVectors ++ that.leafVectors, this.innerVectors ++ that.innerVectors)
  }
}

case class HSMWeightMatrix[T](leafVectors: Map[T, HSMTargetValue],
                              innerVectors: Map[String, DenseVector]) extends WeightMatrix {
  def ++ (that: HSMWeightMatrix[T]): HSMWeightMatrix[T] = {
    HSMWeightMatrix(this.leafVectors ++ that.leafVectors, this.innerVectors ++ that.innerVectors)
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
}

class ContextEmbedder[T: ClassTag: typeinfo.TypeInformation]
  extends Embedder[Context[T], HSMWeightMatrix[T]] {

  val numberOfIterations: Int = parameters(Iterations)
  val minTargetCount: Int = parameters(TargetCount)
  val batchSize: Int = parameters(BatchSize)
  val vectorSize: Int = parameters(VectorSize)
  val learningRate: Double = parameters(LearningRate)

  def optimize(data: DataSet[Context[T]],
               initialWeights: Option[DataSet[HSMWeightMatrix[T]]]): DataSet[HSMWeightMatrix[T]] = {

    val initialWeightsDS: DataSet[HSMWeightMatrix[T]] = createInitialWeightsDS(initialWeights, data)

    initialWeightsDS.iterate(numberOfIterations) {
      weights => FormVectors(data, weights, learningRate)
    }
  }

  def createInitialWeightsDS(initialWeights: Option[DataSet[HSMWeightMatrix[T]]],
                              data: DataSet[Context[T]])
  : DataSet[HSMWeightMatrix[T]] = initialWeights match {
    case Some(weightMatrix) => weightMatrix //extendHSMWeightMatrix(data, weightMatrix)
    case None => formHSMWeightMatrix(data)
  }

  private def extendHSMWeightMatrix(data: DataSet[Context[T]], weights: DataSet[HSMWeightMatrix[T]])
  : DataSet[HSMWeightMatrix[T]] = {
    val env = data.getExecutionEnvironment

    val targets = data
      .map(x => (x.target, 1))
      .groupBy(0).sum(1)

    val existing = weights
      .flatMap(x => x.leafVectors.map(x => x._1 -> x._2.count))

    val combined = targets.union(existing)
      .groupBy(0).sum(1)
      .filter(_._2 >= minTargetCount)

    val softMaxTree = env.fromElements(HuffmanBinaryTree.tree(combined.collect()))

    val leafMap = targets
      .mapWithBcVariable(softMaxTree) {
        (target, softMaxTree) => {
          val code = HuffmanBinaryTree.encode(softMaxTree, target._1)
          val path = HuffmanBinaryTree.path(code)
          target._1 -> HSMTargetValue(
            DenseVector.apply(
              Array.fill(vectorSize)((scala.math.random - 0.5f) / vectorSize)),
            target._2,
            code,
            path
          )
        }
      }

    val innerMap = leafMap
      .map(l => l._2.path).flatMap(x => x)
      .distinct()
      .map(x => x -> DenseVector.zeros(vectorSize))

    val localWeights = weights.collect().head

    env.fromElements(
      HSMWeightMatrix(
        leafMap.collect().toMap ++ localWeights.leafVectors, innerMap.collect().toMap))
  }

  private def formHSMWeightMatrix(data: DataSet[Context[T]])
  : DataSet[HSMWeightMatrix[T]] = {
    val env = data.getExecutionEnvironment

    val targets = data
      .map(x => (x.target, 1))
      .groupBy(0).sum(1)
      .filter(_._2 >= minTargetCount)

    val softMaxTree = env.fromElements(HuffmanBinaryTree.tree(targets.collect()))

    val leafMap = targets
      .mapWithBcVariable(softMaxTree) {
        (target, softMaxTree) => {
          val code = HuffmanBinaryTree.encode(softMaxTree, target._1)
          val path = HuffmanBinaryTree.path(code)
          target._1 -> HSMTargetValue(
            DenseVector.apply(
              Array.fill(vectorSize)((scala.math.random - 0.5f) / vectorSize)),
            target._2,
            code,
            path
          )
        }
      }.map(x => HSMWeightMatrix(Map(x), Map.empty[String, DenseVector]))

    val innerMap = leafMap
      .flatMap(l => l.leafVectors.values)
      .flatMap(x => x.path)
      .distinct()
      .map(x =>
        HSMWeightMatrix(Map.empty[T, HSMTargetValue], Map(x -> DenseVector.zeros(vectorSize))))

    leafMap.union(innerMap).reduce(_ ++ _)
  }

  private def FormVectors(data: DataSet[Context[T]],
                       weights: DataSet[HSMWeightMatrix[T]],
                       learningRate: Double)
  : DataSet[HSMWeightMatrix[T]] = {

    val leafTargetVals = weights
      .flatMap(x => x.leafVectors)

    val localWeights = weights
      .map(x =>
        LocalWeightMatrix(
          x.leafVectors.map(y => y._1 -> y._2.vector),
          x.innerVectors))

    val leafPaths = weights
      .flatMap(x => x.leafVectors)
      .flatMap(x => x._2.path.map(y => x._1 -> y))

    val trainingSets = data
      .map(x => x -> x.target)
      .coGroup(leafPaths).where(1).equalTo(0) {
      (context, path) =>
        val p = path.map(x => x._2)
        context.map(x => HSMTrainingSet(x._1.context.to[Seq], p.to[Seq]))
    }.flatMap(x => x)
    .filter(x => x.inner.nonEmpty)
    .zipWithUniqueId.map(x => Seq(x._2) -> x._1 % batchSize)

    val learnedWeights = trainingSets
      .groupBy(1)
      .reduce((a,b) => a._1 ++ b._1 -> a._2)
      .map(_._1)
      .mapWithBcVariable(localWeights)(broadcastTrain)

    val learnedInnerWeights = learnedWeights
      .flatMap(x => x.innerVectors.toSeq)
      .groupBy(_._1)
      .reduceGroup(learnedVecs => {
        learnedVecs.reduce((a,b) => {
          val aVec = a._2
          val bVec = b._2
          BLAS.axpy(1, aVec, bVec)
          (a._1, bVec)
        })
      })
      .map(x => HSMWeightMatrix[T](Map.empty, Map(x)))
      .reduce(_ ++ _)

    val learnedLeafWeights = learnedWeights
      .flatMap(x => x.leafVectors)
      .groupBy(_._1)
      .reduceGroup(learnedVecs => {
        learnedVecs.reduce((a,b) => {
          val aVec = a._2
          val bVec = b._2
          BLAS.axpy(1, aVec, bVec)
          (a._1, bVec)
        })
      })
      .join(leafTargetVals).where(0).equalTo(0)
      .map(x => HSMWeightMatrix[T](Map(x._2._1 -> x._2._2.copy(vector = x._1._2)), Map.empty))
      .reduce(_ ++ _)

    weights.union(learnedLeafWeights).union(learnedInnerWeights)
      .reduce((a,b) => a ++ b)
  }

  private def broadcastTrain(context: Seq[HSMTrainingSet[T]],
                             weights: LocalWeightMatrix[T])
  : LocalWeightMatrix[T] = train(context, weights, None)

  private def train(context: Seq[HSMTrainingSet[T]],
                    weights: LocalWeightMatrix[T],
                    alpha: Option[Double])
  : LocalWeightMatrix[T] = context match {
    case trainingSet :: tail =>
      val localAlpha = alpha.getOrElse(learningRate)
      val leafVectors = trainingSet.leaf.flatMap(x => weights.leafVectors.get(x) match {
        case Some(vector) => Some(x -> vector)
        case None => None
      })
      val innerVectors = trainingSet.inner.flatMap(x => weights.innerVectors.get(x) match {
        case Some(vector) => Some(x -> vector)
        case None => None
      })
      val decayedAlpha =  localAlpha * 1 - (1 / (tail.size + 1))
      val hiddenVector = DenseVector.zeros(vectorSize)
      val updatedWeights =
        trainOnContext(leafVectors, innerVectors, hiddenVector, weights, decayedAlpha)
      train(tail, weights ++ updatedWeights, Some(decayedAlpha))
    case Nil => weights
  }

  private def trainOnContext(leafVectors: Seq[(T, DenseVector)],
                             innerVectors: Seq[(String, DenseVector)],
                             hiddenLayer: DenseVector,
                             weights: LocalWeightMatrix[T],
                             learningRate: Double)
  : LocalWeightMatrix[T] = leafVectors match {
    case leaf :: tail =>
      val (updatedInner, updatedLeaf, updatedHidden) =
        trainOnWindow(
          innerVectors, Seq.empty[(String, DenseVector)], leaf._2, hiddenLayer, learningRate)
      //learn weights hidden -> input
      BLAS.axpy(1.0, updatedHidden, updatedLeaf)
      val updatedWeights = weights ++ LocalWeightMatrix(
        Map(leaf._1 -> updatedLeaf), updatedInner.toMap)
          trainOnContext(tail, updatedInner, updatedHidden, updatedWeights, learningRate)
    case Nil => weights
  }

  private def trainOnWindow(inputVectors: Seq[(String, DenseVector)],
                            outputVectors: Seq[(String, DenseVector)],
                            leaf: DenseVector,
                            hidden: DenseVector,
                            learningRate: Double)
  : (Seq[(String, DenseVector)], DenseVector, DenseVector) = inputVectors match {
      case inner::tail =>
        val forwardPass = BLAS.dot(leaf, inner._2)
        val nonLinearity = BreezeNumerics.sigmoid(forwardPass)
        val gradient = (1 - inner._1.reverse.head.toInt - nonLinearity) * learningRate
        //axpy works on vectors in place
        //backprop from output -> hidden
        BLAS.axpy(gradient, inner._2, hidden)
        //learn weights hidden -> output
        BLAS.axpy(gradient, leaf, inner._2)
        trainOnWindow(tail, outputVectors :+ inner, leaf, hidden, learningRate)
      case Nil => (outputVectors, leaf, hidden)
    }
}
