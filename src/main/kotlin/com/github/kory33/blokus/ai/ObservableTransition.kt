package com.github.kory33.blokus.ai

import org.deeplearning4j.rl4j.space.Encodable
import org.deeplearning4j.rl4j.space.ObservationSpace

data class ObservableTransition<out A, O : Encodable>(private val observations: List<O>,
                                                      val action: A,
                                                      val reward: Double,
                                                      val isTerminal: Boolean,
                                                      val nextObservation: O,
                                                      private val observationSpace: ObservationSpace<O>) {
    val observationHistory = ObservationHistory(observations, observationSpace)
    val appended = observationHistory.getAppendedHistory(nextObservation)
}