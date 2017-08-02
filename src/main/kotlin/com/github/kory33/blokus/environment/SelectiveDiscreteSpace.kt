package com.github.kory33.blokus.environment

import org.deeplearning4j.rl4j.space.DiscreteSpace
import org.nd4j.linalg.api.ndarray.INDArray

abstract class SelectiveDiscreteSpace<in S> (size: Int) : DiscreteSpace(size) {
    abstract fun computeActionAvailability(state: S) : INDArray
}