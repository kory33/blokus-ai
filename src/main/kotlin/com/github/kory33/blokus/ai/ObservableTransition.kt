package com.github.kory33.blokus.ai

import org.deeplearning4j.rl4j.space.ObservationSpace

data class ObservableTransition<out A, out O>(val observation: O,
                                              val action: A,
                                              val reward: Double,
                                              val isTerminal: Boolean,
                                              val nextObservation: O,
                                              private val observationSpace: ObservationSpace<O>)