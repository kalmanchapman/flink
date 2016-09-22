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
        val skipGrams = input
          .flatMap(x =>
            x.zipWithIndex
              .map(z => {
                val window = (scala.math.random * 100 % fitParameters(WindowSize)).toInt
                Context[String](
                  z._1, x.slice(z._2 - window, z._2) ++ x.slice(z._2 +1, z._2 + window))
              }))

        val weights = new ContextEmbedder[String]
          .setIterations(fitParameters(Iterations))
          .setTargetCount(fitParameters(TargetCount))
          .setVectorSize(fitParameters(VectorSize))
          .setLearningRate(fitParameters(LearningRate))
          .optimize(skipGrams, instance.wordVectors)

        instance.wordVectors = Some(weights)
      }
    }
  }

  implicit def words2Vecs = {
    new TransformDataSetOperation[Word2Vec, String, (String, DenseVector)] {
      override def transformDataSet(
        instance: Word2Vec,
        transformParameters: ParameterMap,
        input: DataSet[String]) : DataSet[(String, DenseVector)] = {
          instance.wordVectors match {
            case Some(vectors) =>
              input.mapWithBcVariable(vectors) {
                (t, weights) => weights.leafVectors.get(t) match {
                  case Some(value) => (t, value.vector)
                  case None => (t, DenseVector())
                }
              }
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
