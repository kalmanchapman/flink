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

package org.apache.flink.ml.nlp

import breeze.{numerics => BreezeNumerics}
import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.api.scala._
import org.apache.flink.ml.RichDataSet
import org.apache.flink.ml.math.{BLAS, DenseVector}
import org.apache.flink.ml.optimization.Solver
import org.apache.flink.util.Collector

import scala.collection.immutable.HashMap

/**
  * Implements Word2Vec; a word-embedding algoritm first described
  * by Tomáš Mikolov et al in http://arxiv.org/pdf/1301.3781.pdf
  *
  *
  */

object HuffmanBinaryTree extends Serializable {
  import scala.collection.mutable.PriorityQueue
  private abstract class Tree[+A]

  private type WeightedNode = (Tree[Any], Int)

  implicit def orderByInverseWeight[A <: WeightedNode]: Ordering[A] =
    Ordering.by {
      case (_, weight) => -1 * weight
    }

  private case class Leaf[A](value: A) extends Tree[A]

  private case class Branch[A](left: Tree[A], right: Tree[A]) extends Tree[A]

  // recursively build the binary tree needed to Huffman encode the text
  // not immutable :(
  private def merge(xs: PriorityQueue[WeightedNode]): PriorityQueue[WeightedNode] = {
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
    merge(new PriorityQueue[WeightedNode] ++= xs).toList
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
        if (contains(l, value))
          turn(l, value, code :+ 0)
        else
          turn(r, value, code :+ 1)
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
  * Hierarchical SoftMax is an approach to the softmax classification solver
  * that utilizes a distributed, sequential representation of class output probabilities
  * via a huffman encoding of known classes and a training process that 'learns'
  * the output classes by traversing the inner probabilities of the encoding
  *
  * More on hierarchical softmax: (http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
  * More on softmax: (https://en.wikipedia.org/wiki/Softmax_function)
  */

case class Context[T](target: T, context: Iterable[T])

trait WeightMatrix

case class HSMTargetValue(vector: DenseVector, code: Vector[Int], path: Vector[String])

case class HSMStepValue(key: String, target: Int, vector: DenseVector)

case class HSMWeightMatrix[T](leafVectors: Map[T, HSMTargetValue],
                              innerVectors: Map[String, DenseVector]) extends WeightMatrix

trait TrainingSet

case class HSMTrainingSet[T](leafKeys: Iterable[T],
                             innerKeys: Iterable[(String, Int)]) extends TrainingSet

class ContextClassifier extends Solver[Context, HSMWeightMatrix] {

  val numberOfIterations: Int = 1
  val minTargetCount: Int = 5
  val vectorSize: Int = 100
  val learningRate: Double = 0.015

  def optimize[T](data: DataSet[Context[T]],
                  initialWeights: Option[DataSet[HSMWeightMatrix[T]]]): DataSet[HSMWeightMatrix[T]] = {

    val initialWeightsDS: DataSet[HSMWeightMatrix[T]] = createInitialWeightsDS(initialWeights, data)



  }

  def createInitialWeightsDS[T](initialWeights: Option[DataSet[HSMWeightMatrix[T]]],
                                data: DataSet[Context[T]]
                               ): DataSet[HSMWeightMatrix[T]] = initialWeights match {
    //check on weight set? - expansion on weight set?
    case Some(weightMatrix) => weightMatrix
    case None => formHSMWeightMatrix(data)
  }

  private def formHSMWeightMatrix[T](data: DataSet[Context[T]]): DataSet[HSMWeightMatrix[T]] = {
    val env = data.getExecutionEnvironment

    val targets = data
      .map(x => (x.target, 1)).groupBy(0).sum(1)
      .filter(_._2 >= minTargetCount)

    val softMaxTree = env.fromElements(HuffmanBinaryTree.tree(targets.collect()))

    val leafMap = targets
      .mapWithBcVariable(softMaxTree) {
        (target, softMaxTree) => {
          val code = HuffmanBinaryTree.encode(softMaxTree, target._1)
          val path = HuffmanBinaryTree.path(code)
          target._1 -> HSMTargetValue(
            DenseVector.apply(
              Array.fill(vectorSize)((math.random - 0.5f) / vectorSize)),
            code,
            path
          )
        }
      }

    val innerMap = leafMap
      .map(l => l._2.path).flatMap(x => x)
      .distinct
      .map(x => x -> DenseVector.zeros(vectorSize))

    env.fromElements(HSMWeightMatrix(leafMap.collect.toMap, innerMap.collect.toMap))
  }

  //I would *LOVE* to use a mapPartitionWithBcVariable - but it hasn't been implemented!
  //why is that? well - the abstract Java method returns Void - and scala extensions can't
  //fulfill the requirements of such a method with icky trickery...
  private def SkipGram[T](
                           data: DataSet[Context[T]],
                           weights: DataSet[HSMWeightMatrix[T]],
                           learningRate: Double)
  : DataSet[HSMWeightMatrix[T]] = {
    data
      .mapWithBcVariable(weights)(mapContext)
      .flatMap(x => x)
      .mapPartition((t, collector: Collector[HSMWeightMatrix[T]]) => {
        trainOnPartition(t.toList, weights.collect().head, learningRate)
      })
      .reduce((a,b) => b)
  }

  //simplify this - it's not really a map, it's a filter - but on both the super
  //and sub objects
  private def mapContext[T](data: Context[T],
                            weights: HSMWeightMatrix[T]): Option[HSMTrainingSet[T]] = weights.leafVectors.get(data.target) match {
    case Some(targetValue) =>
      Option(
        HSMTrainingSet(
          data.context.filter(c => weights.leafVectors.contains(c)),
          targetValue.path.zip(targetValue.code)
            .filter(c => weights.innerVectors.contains(c._1))))
    case _ => None
  }

  //should comment the heck out of this, as the magic occurs here
  //we get a 'subset' of leaf vectors, and inner vectors, and we need to loop over each
  //inner vector for each leaf vector - performing calculations and error updates at each step.


  //loops on (target, contextSet) pairs
  private def trainOnPartition[T](contextTrainingSet: List[HSMTrainingSet[T]],
                                  localWeights:HSMWeightMatrix[T],
                                  learningRate: Double)
  : HSMWeightMatrix[T] = contextTrainingSet match {
    case contextSet :: tail => {
      val innerVectors = contextSet.innerKeys.flatMap(k => localWeights.innerVectors.get(k._1) match {
        case Some(innerVector) => Option(HSMStepValue(k._1, k._2, innerVector))
        case _ => None
      }).toList
      val leafVectors = contextSet.leafKeys.flatMap(k => localWeights.leafVectors.get(k) match {
        case Some(leafVector) => Option(k -> leafVector)
        case _ => None
      }).toList
      val hiddenVector = DenseVector.zeros(vectorSize)

      val partialWeights = trainOnContext(leafVectors, innerVectors, hiddenVector,
        HSMWeightMatrix(Map.empty[T, HSMTargetValue], Map.empty[String, DenseVector]), learningRate)

      val adjustedWeights =
        HSMWeightMatrix(localWeights.leafVectors ++ partialWeights.leafVectors, localWeights.innerVectors ++ partialWeights.innerVectors)
      trainOnPartition(tail, adjustedWeights, learningRate)
    }
    case Nil => localWeights
  }

  //loops on (target, context) pairs
  private def trainOnContext[T](leafVectors: List[(T, HSMTargetValue)],
                                innerVectors: List[HSMStepValue],
                                hiddenLayer: DenseVector,
                                partialWeights: HSMWeightMatrix[T],
                                learningRate: Double): HSMWeightMatrix[T] = leafVectors match {
    case leaf :: tail => {
      val (updatedInner, updatedLeaf, updatedHidden) = trainOnWindow(innerVectors, List.empty[HSMStepValue], leaf._2.vector, hiddenLayer, learningRate)
      BLAS.axpy(1.0, updatedHidden, updatedLeaf)
      val updatedWeights = HSMWeightMatrix(
        partialWeights.leafVectors ++ Map[T, HSMTargetValue](leaf._1 -> leaf._2.copy(vector = updatedLeaf)),
        partialWeights.innerVectors ++ updatedInner.map(i => i.key -> i.vector))
      trainOnContext(tail, updatedInner, updatedHidden, updatedWeights, learningRate)
    }
    case Nil => partialWeights
  }

  //loops on (targetVec, context) vector pairs in window
  private def trainOnWindow(inputVectors: List[HSMStepValue],
                            outputVectors: List[HSMStepValue],
                            leaf: DenseVector,
                            hidden: DenseVector,
                            learningRate: Double)
  : (List[HSMStepValue], DenseVector, DenseVector) = inputVectors match {
    case inner::tail =>
      val forwardPass = BLAS.dot(leaf, inner.vector)
      val nonLinearity = BreezeNumerics.sigmoid(forwardPass)
      val gradient = (1 - inner.target - nonLinearity) * learningRate
      //axpy works on vectors in place
      //backprop from output -> hidden
      BLAS.axpy(gradient, inner.vector, hidden)
      //learn weights hidden -> output
      BLAS.axpy(gradient, leaf, inner.vector)
      trainOnWindow(tail, inner :: outputVectors, leaf, hidden, learningRate)
    case Nil => (outputVectors, leaf, hidden)
  }
}
