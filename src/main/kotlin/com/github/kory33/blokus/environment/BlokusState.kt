package com.github.kory33.blokus.environment

import org.deeplearning4j.gym.StepReply

class BlokusState {
    constructor() {
        TODO("Not implemented due to Engine API mismatch")
    }

    /**
     * Get A(s) where s = this
     */
    fun getPossibleActions() : Set<BlokusAction> {
        TODO("Not implemented due to Engine API mismatch")
    }

    fun hasGameFinished() : Boolean {
        TODO("Not implemented due to Engine API mismatch")
    }

    fun step(action : BlokusAction?) : StepReply<BlokusState> {
        TODO("Not implemented due to Engine API mismatch")
    }
}