package com.github.kory33.blokus.ai

import com.github.kory33.blokus.environment.SelectiveDiscreteSpace
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.network.dqn.DQNFactory
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.space.Encodable
import org.deeplearning4j.rl4j.util.DataManager

class QLearningSelectiveDiscreteDense<O : Encodable, AS : SelectiveDiscreteSpace<O>>
        (mdp : MDP<O, Int, AS>,
         dqn : IDQN,
         conf : QLearning.QLConfiguration,
         dataManager: DataManager)
    : QLearningSelectiveDiscrete<O, AS>(mdp, dqn, conf, dataManager, conf.epsilonNbStep) {

    constructor (mdp: MDP<O, Int, AS>,
                 factory: DQNFactory,
                 conf: QLearning.QLConfiguration,
                 dataManager: DataManager)
            : this(mdp, factory.buildDQN(mdp.observationSpace.shape, mdp.actionSpace.size), conf, dataManager)

    constructor (mdp: MDP<O, Int, AS>,
                 netConf: DQNFactoryStdDense.Configuration,
                 conf: QLearning.QLConfiguration,
                 dataManager: DataManager)
            : this(mdp, DQNFactoryStdDense(netConf), conf, dataManager)
}
