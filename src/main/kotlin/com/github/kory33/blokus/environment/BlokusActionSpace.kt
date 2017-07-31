package com.github.kory33.blokus.environment

import com.github.kory33.blokus.util.SetUtil
import org.deeplearning4j.rl4j.space.ActionSpace

class BlokusActionSpace(state: BlokusState) : ActionSpace<BlokusAction> {
    private val availableActions : Set<BlokusAction> = state.getPossibleActions()

    override fun getSize(): Int {
        return this.availableActions.size
    }

    override fun randomAction(): BlokusAction {
        return SetUtil.chooseRandomlyFrom(this.availableActions)
    }

    override fun noOp(): BlokusAction? {
        return null
    }

    override fun encode(action: BlokusAction?): Any {
        return action.toString()
    }
}