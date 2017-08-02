package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusPlacementSpace
import com.github.kory33.blokus.game.IBlokusPlayer
import com.github.kory33.blokus.game.color.PlayerColor
import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.space.DiscreteSpace
import org.deeplearning4j.rl4j.space.ObservationSpace

/**
 * Blokus game represented as Markov Decision Process.
 *
 * Before starting training on this mdp, an "exploiting adversary" must be set
 * to the corresponding field. Otherwise members in this object remains uninitialized, possibly resulting
 * in an exception.
 */
class BlokusMDP(private val playerColor: PlayerColor) : MDP<BlokusObservation, Int, DiscreteSpace> {
    lateinit var state: BlokusState
    private var exploitingAdversary: IBlokusPlayer? = null
        set(newAdversary) {
            field = newAdversary
            reset()
        }

    override fun reset(): BlokusObservation {
        this.state = BlokusState(exploitingAdversary!!, playerColor)
        return this.state.observation
    }

    override fun close() {}

    override fun step(action: Int): StepReply<BlokusObservation> {
        return this.state.step(BlokusPlacementSpace.PLACEMENT_LIST[action])
    }

    override fun getActionSpace(): DiscreteSpace = ACTION_SPACE

    override fun isDone(): Boolean = this.state.hasGameFinished()

    override fun newInstance(): MDP<BlokusObservation, Int, DiscreteSpace> = BlokusMDP(this.playerColor)

    override fun getObservationSpace(): ObservationSpace<BlokusObservation> = OBSERVATION_SPACE

    companion object {
        val OBSERVATION_SPACE = BlokusObservationSpace()
        val ACTION_SPACE = BlokusActionSpace()
    }
}