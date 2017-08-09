package com.github.kory33.blokus.ai

import com.github.kory33.blokus.environment.space.SelectiveDiscreteSpace
import org.deeplearning4j.rl4j.space.Encodable

interface ISelectiveObservation : Encodable {
    fun getFilteringActionSpace() : SelectiveDiscreteSpace
}