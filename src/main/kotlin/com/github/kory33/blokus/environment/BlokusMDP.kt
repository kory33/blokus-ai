package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.IBlokusPlayer
import com.github.kory33.blokus.game.color.PlayerColor
import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.space.ObservationSpace

/**
 * Blokus game represented as Markov Decision Process.
 *
 * Before starting training on this mdp, an "exploiting adversary" must be set
 * to the corresponding field. Otherwise members in this object remains uninitialized, possibly resulting
 * in an exception.
 */
class BlokusMDP(private val playerColor: PlayerColor) : MDP<BlokusState, BlokusAction, BlokusActionSpace> {
    lateinit var state: BlokusState
    private var exploitingAdversary: IBlokusPlayer? = null
        set(newAdversary) {
            field = newAdversary
            reset()
        }

    override fun reset(): BlokusState {
        val newState = BlokusState(exploitingAdversary!!, playerColor)
        this.state = newState
        return newState
    }

    override fun close() {}

    override fun step(action: BlokusAction?): StepReply<BlokusState> {
        return this.state.step(action)
    }

    override fun getActionSpace(): BlokusActionSpace = BlokusActionSpace(this.state)

    override fun isDone(): Boolean = this.state.hasGameFinished()

    override fun newInstance(): MDP<BlokusState, BlokusAction, BlokusActionSpace>
            = BlokusMDP(this.playerColor)

    override fun getObservationSpace(): ObservationSpace<BlokusState> = OBSERVATION_SPACE

    companion object {
        val OBSERVATION_SPACE : ObservationSpace<BlokusState> = BlokusObservationSpace()
    }
}