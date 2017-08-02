package com.github.kory33.blokus.environment

import org.deeplearning4j.rl4j.space.DiscreteSpace
import org.nd4j.linalg.api.ndarray.INDArray

abstract class SelectiveDiscreteSpace (size: Int) : DiscreteSpace(size) {
    /**
     * Computes a vector with elements 1 if the action corresponding to the
     * index of the element is valid, 0 if not.
     */
    abstract fun computeActionAvailability() : INDArray
}