package com.github.kory33.blokus.environment.space

import org.deeplearning4j.rl4j.space.DiscreteSpace
import org.nd4j.linalg.api.ndarray.INDArray

abstract class SelectiveDiscreteSpace (size: Int) : DiscreteSpace(size) {
    /**
     * Computes a vector with elements 1 if the action corresponding to the
     * index of the element is valid, 0 if not.
     */
    abstract fun computeActionAvailability() : INDArray

    /**
     * Computes a list containing all the valid actions
     */
    abstract fun getAvailableActions() : List<Int>

    abstract fun maxQValueAction(qVector : INDArray) : Int

    override fun randomAction(): Int {
        val availableActions = getAvailableActions()
        return availableActions[rd.nextInt(availableActions.size)]
    }

    fun maxQValue(qVector: INDArray): Double {
        return qVector.getDouble(maxQValueAction(qVector))
    }

    fun filterQValues(qVector: INDArray): INDArray {
        return qVector.muliRowVector(computeActionAvailability())
    }
}