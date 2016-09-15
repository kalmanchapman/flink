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

package org.apache.flink.ml.feature

import breeze.linalg.{DenseMatrix => BreezeMatrix}
import org.apache.flink.api.common.functions.{RichMapFunction, RichMapPartitionFunction}
import org.apache.flink.api.scala._
import org.apache.flink.ml.optimization.IterativeSolver

import scala.collection.immutable.HashMap

/**
  * Implements Word2Vec; a word-embedding algoritm first described
  * by Tomáš Mikolov et al in http://arxiv.org/pdf/1301.3781.pdf
  *
  * blah blah blah, let's come back to this
  */

case class Word (
  var value: String,
  var count: Int,
  var vector: BreezeMatrix[Double],
  var code: Vector[Int]
)

case class Vocabulary(
  var lexicon: Vector[Word],
  var lexiCount: Int,
  var lexiHash: Map[String, Int],
  var softmaxHash: Map[String, Int]
)

object HuffmanBinaryTree {
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

class Word2Vec {

}

object Word2Vec {

  //should be constant
  val BC_WORD_VECS = "broadcast_word_vectors"
  val BC_SOFTMAX_VECS = "broadcast_softmax_vectors"
  val BC_LEXI_HASH = "broadcast_lexi_hash"
  val BC_SOFTMAX_HASH = "broadcast_softmax_hash"
  val BC_LEXICON = "broadcast_lexicon"

  //should be settable
  val learningRate = 0.025
  val wordCount = 5
  val vectorLength = 100
  val maxSentenceLength = 1000

  def train(dataSet: DataSet[Iterable[String]]): Vocabulary = {

    val env = dataSet.getExecutionEnvironment

    val vocabulary = buildVocab(dataSet)

    val wordVecsGlobal = vocabulary.lexicon
      .map(w => w.vector)
      .reduce(BreezeMatrix.vertcat(_, _))

    val softmaxVecsGlobal = BreezeMatrix().apply(vectorLength, vocabulary.lexiCount)

    val alpha = learningRate

    //convert sentences into references to vocabulary.lexicon
    val sentences =

    //iterator?

    val modifiedWeights =



  }

  private def buildVocab(dataSet: DataSet[Iterable[String]]): Vocabulary = {

    val env = dataSet.getExecutionEnvironment

    //decomposes the input corpus into a localized wordcount
    val weightedLexicon = dataSet
      .flatMap(x => x)
      .map(w => (w, 1))
      .groupBy(0).sum(1)
      .filter(_._2 >= wordCount)
      .collect()

    val lexiCount = weightedLexicon.size

    //forms a huffman tree from the wordcount -
    // other weighting might yield interesting results
    val softmaxTree = HuffmanBinaryTree.tree(weightedLexicon)

    val lexicon = weightedLexicon
      .map(w => Word(
        w._1,
        w._2,
        BreezeMatrix.create(1, vectorLength,
          Array.fill(vectorLength)((math.random - 0.5f) / vectorLength)),
        HuffmanBinaryTree.encode(softmaxTree, w._1)))

    //gives a mapping of word value to index in lexicon
    val lexiHash = lexicon
      .zipWithIndex
      .map(w => HashMap(w._1.value -> w._2))
      .reduce(_ ++ _)

    //gives a mapping of binary tree inner node to
    //an imaginary 'index' of these values - these will
    //map to the weighted vectors between the hidden and output
    //layers of our network - note that each index will be trained
    //proportionate to the number of words at the subtree served by that branch
    //and their frequency
    val pathHash = lexicon
      .map(w => HuffmanBinaryTree.path(w.code))
      .flatMap(p => p)
      .distinct
      .zipWithIndex
      .map(p => HashMap(p._1 -> p._2))
      .reduce(_ ++ _)

    Vocabulary(
      lexicon.to[Vector],
      lexiCount,
      lexiHash,
      pathHash)
  }

  class sentenceToVocabularyMapper
    extends RichMapPartitionFunction[Iterable[String], Vector[Int]] {
      var
  }
}
