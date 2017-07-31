package com.github.kory33.blokus.environment

import org.deeplearning4j.rl4j.space.ObservationSpace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
 * Represents a space of valid observations
 */
class BlokusObservationSpace : ObservationSpace<BlokusState> {

    override fun getName(): String = NAME
    override fun getLow(): INDArray = LOW
    override fun getHigh(): INDArray = HIGH
    override fun getShape(): IntArray = SHAPE

    companion object {
        val NAME = "BlokusObservationSpace"
        /**
         * Shape of an observation
         */
        val SHAPE = arrayOf(12, 12, 3).toIntArray()
        val LOW = Nd4j.zeros(*SHAPE)!!
        val HIGH = Nd4j.zeros(*SHAPE).add(1)!!
    }
}