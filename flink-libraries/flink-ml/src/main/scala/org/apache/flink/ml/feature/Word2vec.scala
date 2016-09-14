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

import org.apache.flink.api.scala.DataSet
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.pipeline.Predictor

import scala.collection.immutable.HashMap
import scala.collection.mutable.PriorityQueue
/**
  * Implements Word2Vec; a word-embedding algoritm first described
  * by Tomáš Mikolov et al in http://arxiv.org/pdf/1301.3781.pdf
  *
  * blah blah blah, let's come back to this
  */

case class Word(
  var value: String,
  var count: Int,
  var vector: DenseVector,
  var code: Vector[Int],
  var path: Vector[String]
)

case class Vocabulary(
  var lexicon: Vector[Word],
  var lexiCount: Int,
  var lexiHash: Map[String, Int],
  var pathHash: Map[String, Int]
)

object HuffmanBinaryTree {

  abstract class Tree[+A]

  type WeightedNode = (Tree[Any], Int)

  implicit def orderByInverseWeight[A <: WeightedNode]: Ordering[A] =
    Ordering.by {
      case (_, weight) => -1 * weight
    }

  case class Leaf[A](value: A) extends Tree[A]

  case class Branch[A](left: Tree[A], right: Tree[A]) extends Tree[A]

  // recursively build the binary tree needed to Huffman encode the text
  // not immutable :(
  def merge(xs: PriorityQueue[WeightedNode]): PriorityQueue[WeightedNode] = {
    if (xs.length == 1) xs
    else {
      val l = xs.dequeue
      val r = xs.dequeue
      val merged = (Branch(l._1, r._1), l._2 + r._2)
      merge(xs += merged)
    }
  }

  //convenience for building and extracting from priority queue
  def merge(xs: Iterable[WeightedNode]): Iterable[WeightedNode] = {
    //form priority queue, build tree, return a list (should have only 1 member)
    merge(new PriorityQueue[WeightedNode] ++= xs).toList
  }

  // recursively search the branches of the tree for the required character
  def contains(tree: Tree[Any], value: Any): Boolean = tree match {
    case Leaf(c) => if (c == value) true else false
    case Branch(l, r) => contains(l, value) || contains(r, value)
  }

  //takes a code and decomposes into visited node identities
  def path(code: Vector[Int]) : Vector[String] = {
    def form(code: Vector[Int], path: Vector[String]): Vector[String] = code match {
      case head +: tail => form(tail, path :+ (path.head + code))
      case IndexedSeq() => path
    }
    form(code, Vector("root"))
  }
  // recursively build the path string required to traverse the tree to the required character
  def encode(tree: Tree[Any], value: Any): (Vector[Int], Vector[String]) = {
    def turn(tree: Tree[Any], value: Any,
             code: Vector[Int], path: Vector[String]): (Vector[Int], Vector[String]) = tree match {
      case Leaf(_) => (code, path)
      case Branch(l, r) =>
        if (contains(l, value))
          turn(l, value, code :+ 0, path :+ code.mkString)
        else
          turn(r, value, code :+ 1, path :+ code.mkString)
    }
    turn(tree, value, Vector.empty[Int], Vector("root"))
  }

  def tree(weightedLexicon: Iterable[(Any, Int)]): Tree[Any] =
    merge(weightedLexicon.map(x => (Leaf(x._1), x._2))).head._1
}

class Word2Vec extends Predictor[Word2Vec] {

}

object Word2Vec {

  //should be settable
  val wordCount = 5
  val vectorLength = 100

  def buildVocab(dataSet: DataSet[Iterable[String]]):Vocabulary = {

    val env = dataSet.getExecutionEnvironment

    val weightedLexicon = dataSet
      .flatMap(x => x)
      .map(w => (w, 1))
      .groupBy(0).sum(1)
      .filter(_._2 >= wordCount)
      .collect()

    val lexiCount = weightedLexicon.size

    val softmaxTree = HuffmanBinaryTree.tree(weightedLexicon)

    val lexiHash = weightedLexicon
      .zipWithIndex
      .map(w => HashMap(w._1._1 -> w._2))
      .reduce(_ ++ _)

    val lexicon = weightedLexicon
      .map(w => (w._1, w._2, HuffmanBinaryTree.encode(softmaxTree, w._1)))
      .map(w => Word(
        w._1,
        w._2,
        DenseVector.apply(
          Array.fill[Double](vectorLength)((math.random - 0.5f) / vectorLength)),
        w._3._1,
        w._3._2))
      .to[Vector]

    val pathHash = lexicon
      .map(w => w.path)
      .distinct
      .zipWithIndex
      .map(p => HashMap(p._1 -> p._2))
      .reduce(_ ++ _)


    Vocabulary(
      lexicon,
      lexiCount,
      lexiHash,
      pathHash)
  }

}
