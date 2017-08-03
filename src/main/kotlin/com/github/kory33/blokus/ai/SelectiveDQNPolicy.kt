package com.github.kory33.blokus.ai

import com.github.kory33.blokus.environment.SelectiveDiscreteSpace
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.policy.DQNPolicy
import org.deeplearning4j.rl4j.space.Encodable
import org.nd4j.linalg.api.ndarray.INDArray

class SelectiveDQNPolicy<O : Encodable, AS : SelectiveDiscreteSpace>(private val mdp : MDP<O, Int, AS>,
                                                                     private val dqn : IDQN) : DQNPolicy<O>(dqn) {
    override fun nextAction(input: INDArray): Int {
        return mdp.actionSpace.maxQValueAction(dqn.output(input))
    }
}