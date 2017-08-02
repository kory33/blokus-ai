package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.IBlokusPlayer
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
class BlokusMDP(private val playerColor: PlayerColor) : MDP<BlokusState, Int, BlokusActionSpace> {
    lateinit var state: BlokusState
    private var exploitingAdversary: IBlokusPlayer? = null

    fun setAdversary(adversary : IBlokusPlayer) {
            exploitingAdversary = adversary
            reset()
        }

    override fun reset(): BlokusState {
        this.state = BlokusState(exploitingAdversary!!, playerColor)
        return this.state
    }

    override fun close() {}

    override fun step(action: Int): StepReply<BlokusState> {
        return this.state.step(BlokusActionSpace.getPlacementCorrespondingToIndex(action)!!)
    }

    override fun getActionSpace() = BlokusActionSpace(state)

    override fun getObservationSpace() = OBSERVATION_SPACE

    override fun isDone() = this.state.hasGameFinished()

    override fun newInstance() = BlokusMDP(this.playerColor)

    companion object {
        val OBSERVATION_SPACE = BlokusObservationSpace()
    }
}