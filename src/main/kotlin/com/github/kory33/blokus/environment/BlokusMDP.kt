package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.IBlokusPlayer
import com.github.kory33.blokus.game.color.PlayerColor
import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.space.ObservationSpace

/**
 * Blokus game represented as Markov Decision Process.
 */
class BlokusMDP(neuralNetFetchable: NeuralNetFetchable<IDQN>,
                private val exploitingAdversary: IBlokusPlayer,
                private val playerColor: PlayerColor) : MDP<BlokusState, BlokusAction, BlokusActionSpace> {
    private var state: BlokusState

    private var neuralNetFetchable: NeuralNetFetchable<IDQN>
        get() = field

    init {
        this.neuralNetFetchable = neuralNetFetchable
        this.state = BlokusState(exploitingAdversary, playerColor)
    }

    override fun reset(): BlokusState {
        this.state = BlokusState(exploitingAdversary, playerColor)
        return this.state
    }

    override fun close() {}

    override fun step(action: BlokusAction?): StepReply<BlokusState> {
        return this.state.step(action)
    }

    override fun newInstance(): MDP<BlokusState, BlokusAction, BlokusActionSpace>
            = BlokusMDP(this.neuralNetFetchable, this.exploitingAdversary, this.playerColor)

    override fun getObservationSpace(): ObservationSpace<BlokusState> = OBSERVATION_SPACE

    override fun getActionSpace(): BlokusActionSpace = BlokusActionSpace(this.state)

    override fun isDone(): Boolean = this.state.hasGameFinished()
    
    companion object {
        val OBSERVATION_SPACE : ObservationSpace<BlokusState> = BlokusObservationSpace()
    }
}