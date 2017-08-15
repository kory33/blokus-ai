package com.github.kory33.blokus.ai

import org.deeplearning4j.rl4j.space.Encodable
import org.deeplearning4j.rl4j.space.ObservationSpace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class ObservationHistory<O : Encodable>(list: List<O>,
                                        private val observationSpace: ObservationSpace<O>) : List<O> by list {

    private val ndArrayOfHistory = this.map { Nd4j.create(it.toArray(), observationSpace.shape) }.toTypedArray()

    fun getAppendedHistory(observation: O) : ObservationHistory<O> {
        val resultList = List(this.size, { index ->
            when(index) {
                0 -> observation
                else -> this[index - 1]
            }
        })
        return ObservationHistory(resultList, observationSpace)
    }

    private fun concat() : INDArray {
        val concatenated = Nd4j.concat(0, *ndArrayOfHistory)
        if (concatenated.shape().size > 2)
            concatenated.muli(1 / 256f)
        return concatenated
    }

    val concatinated = this.concat()
}