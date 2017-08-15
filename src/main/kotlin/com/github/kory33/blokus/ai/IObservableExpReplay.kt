package com.github.kory33.blokus.ai

import org.deeplearning4j.rl4j.space.Encodable
import java.util.ArrayList

interface IObservableExpReplay<A, O : Encodable> {
    /**
     * @return a batch of uniformly sampled transitions
     */
    fun getBatch(): ArrayList<ObservableTransition<A, O>>

    /**
     * @param transition a new transition to store
     */
    fun store(transition: ObservableTransition<A, O>)
}