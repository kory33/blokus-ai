package com.github.kory33.blokus.ai

import com.github.kory33.blokus.environment.space.SelectiveDiscreteSpace
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.util.DataManager

class QLearningSelectiveHistorylessDiscreteDense<O : ISelectiveObservation, AS : SelectiveDiscreteSpace>
        (mdp : MDP<O, Int, AS>,
         dqn : IDQN,
         conf : QLearning.QLConfiguration,
         dataManager: DataManager)
    : QLearningSelectiveHistorylessDiscrete<O, AS>(mdp, dqn, conf, dataManager, conf.epsilonNbStep)
