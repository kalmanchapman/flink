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

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.flink.ml._
import org.apache.flink.ml.common.{Parameter, ParameterMap}
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.optimization.{Context, ContextEmbedder, HSMWeightMatrix}
import org.apache.flink.ml.pipeline.{FitOperation, TransformDataSetOperation, Transformer}

import scala.reflect.ClassTag

/**
  * Created by kal on 9/22/16.
  */
class Word2Vec extends Transformer[Word2Vec] {
  import Word2Vec._

  private [nlp] var wordVectors:
    Option[DataSet[HSMWeightMatrix[String]]] = None

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

  def setWindowSize(windowSize: Int): this.type = {
    parameters.add(WindowSize, windowSize)
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

object Word2Vec {
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

  case object WindowSize extends Parameter[Int] {
    val defaultValue = Some(10)
  }

  case object BatchSize extends Parameter[Int] {
    val defaultValue = Some(1000)
  }

  case object Seed extends Parameter[Long] {
    val defaultValue = Some(scala.util.Random.nextLong)
  }

  def apply(): Word2Vec = {
    new Word2Vec()
  }

  implicit def learnWordVectors[T <: Iterable[String]] = {
    new FitOperation[Word2Vec, T] {
      override def fit(
        instance: Word2Vec,
        fitParameters: ParameterMap,
        input: DataSet[T])
      : Unit = {
        val resultingParameters = instance.parameters ++ fitParameters
        
        val skipGrams = input
          .flatMap(x =>
            x.zipWithIndex
              .map(z => {
                val window = (scala.math.random * 100 % resultingParameters(WindowSize)).toInt
                Context[String](
                  z._1, x.slice(z._2 - window, z._2) ++ x.slice(z._2 +1, z._2 + window))
              }))

        val weights = new ContextEmbedder[String]
          .setIterations(resultingParameters(Iterations))
          .setTargetCount(resultingParameters(TargetCount))
          .setVectorSize(resultingParameters(VectorSize))
          .setLearningRate(resultingParameters(LearningRate))
          .setBatchSize(resultingParameters(BatchSize))
          .setSeed(resultingParameters(Seed))
          .createInitialWeightsDS(instance.wordVectors, skipGrams)

        instance.wordVectors = Some(weights)
      }
    }
  }

  implicit def words2Vecs[T <: Iterable[String]] = {
    new TransformDataSetOperation[Word2Vec, T, (String, Vector[Double])] {
      override def transformDataSet(instance: Word2Vec,
                                    transformParameters: ParameterMap,
                                    input: DataSet[T]): DataSet[(String, Vector[Double])] = {
        val resultingParameters = instance.parameters ++ transformParameters
        
        instance.wordVectors match {
          case Some(vectors) =>
            val skipGrams = input
              .flatMap(x =>
                x.zipWithIndex
                  .map(z => {
                    val window = (scala.math.random * 100 % resultingParameters(WindowSize)).toInt
                    Context[String](
                      z._1, x.slice(z._2 - window, z._2) ++ x.slice(z._2 + 1, z._2 + window))
                  }))

            val learnedVectors = new ContextEmbedder[String]
              .setIterations(resultingParameters(Iterations))
              .setTargetCount(resultingParameters(TargetCount))
              .setVectorSize(resultingParameters(VectorSize))
              .setLearningRate(resultingParameters(LearningRate))
              .setBatchSize(resultingParameters(BatchSize))
              .setSeed(resultingParameters(Seed))
              .optimize(skipGrams, instance.wordVectors)

            learnedVectors
              .flatMap(_.fetchVectors)
          case None =>
            throw new RuntimeException(
              """
                   the Word2Vec has not been trained on any words!
                   you must fit the transformer to a corpus of text before
                   any context can be extracted!
                """)
          }
      }
    }
  }
}
