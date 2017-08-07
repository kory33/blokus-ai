package com.github.kory33.blokus.environment

import com.github.kory33.blokus.ai.ExploitingMLN
import com.github.kory33.blokus.environment.space.BlokusActionSpace
import com.github.kory33.blokus.environment.space.BlokusObservationSpace
import com.github.kory33.blokus.game.color.PlayerColor
import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.mdp.MDP

/**
 * Blokus game represented as Markov Decision Process.
 *
 * Before starting training on this mdp, an "exploiting adversary" must be set
 * to the corresponding field. Otherwise members in this object remains uninitialized, possibly resulting
 * in an exception.
 */
class BlokusMDP(private val playerColor: PlayerColor, private var exploitingMLN: ExploitingMLN<BlokusActionSpace>)
        : MDP<BlokusState, Int, BlokusActionSpace> {
    var state = BlokusState(exploitingMLN, playerColor)
    var actionOnReset : ((BlokusState) -> Unit)? = null
    var actionSpaceCache : BlokusActionSpace? = null

    override fun reset(): BlokusState {
        this.actionOnReset?.invoke(state)
        this.actionSpaceCache = null
        this.state = BlokusState(exploitingMLN, playerColor)
        return this.state
    }

    override fun close() {}

    override fun step(action: Int): StepReply<BlokusState> {
        this.actionSpaceCache = null
        return this.state.step(BlokusActionSpace.getPlacementCorrespondingToIndex(action)!!)
    }

    override fun getActionSpace() : BlokusActionSpace {
        if (actionSpaceCache == null) {
            actionSpaceCache = state.getActionSpace()
        }
        return actionSpaceCache!!
    }

    override fun getObservationSpace() = OBSERVATION_SPACE

    override fun isDone() = this.state.hasGameFinished()

    override fun newInstance() = BlokusMDP(this.playerColor, this.exploitingMLN)

    companion object {
        val ACTION_SPACE_SIZE = BlokusActionSpace.size
        val OBSERVATION_SPACE = BlokusObservationSpace()
    }
}