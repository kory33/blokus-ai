package com.github.kory33.blokus.environment

import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.space.ObservationSpace

/**
 * Blokus game represented as Markov Decision Process.
 */
class BlokusMDP(var neuralNetFetchable: NeuralNetFetchable<IDQN>?) : MDP<BlokusState, BlokusAction, BlokusActionSpace> {
    private var state: BlokusState = BlokusState()


    override fun reset(): BlokusState {
        this.state = BlokusState()
        return this.state
    }

    override fun close() {}

    override fun step(action: BlokusAction?): StepReply<BlokusState> {
        return this.state.step(action)
    }

    override fun newInstance(): MDP<BlokusState, BlokusAction, BlokusActionSpace> = BlokusMDP(this.neuralNetFetchable)

    override fun getObservationSpace(): ObservationSpace<BlokusState> = OBSERVATION_SPACE

    override fun getActionSpace(): BlokusActionSpace = BlokusActionSpace(this.state)

    override fun isDone(): Boolean = this.state.hasGameFinished()
    
    companion object {
        val OBSERVATION_SPACE : ObservationSpace<BlokusState> = BlokusObservationSpace()
    }
}