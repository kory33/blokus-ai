package com.github.kory33.blokus.test

import com.github.kory33.blokus.ai.ExploitingMLN
import com.github.kory33.blokus.ai.QLearningSelectiveDiscreteDense
import com.github.kory33.blokus.environment.BlokusMDP
import com.github.kory33.blokus.game.color.PlayerColor
import com.github.kory33.blokus.test.util.BoardUtil
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.network.dqn.DQN
import org.deeplearning4j.rl4j.util.DataManager
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage


val blokusQLConfig = QLearning.QLConfiguration(
        123456789, //Random seed
        10000, //Max step By epoch
        25000, //Max step
        15000, //Max size of experience replay
        64, //size of batches
        100, //target update (hard)
        0, //num step noop warmup
        1.00, //reward scaling
        0.99, //gamma
        10.0, //td-error clipping
        0.2f, //min epsilon
        1000, //num step for eps greedy anneal
        true   //double DQN
)

fun main(args: Array<String>) {
    blokusDQN()
}

fun blokusDQN() {
    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()

    val manager = DataManager()
    val network = getBlokusMLN(BlokusMDP.OBSERVATION_SPACE.shape, BlokusMDP.ACTION_SPACE_SIZE)

    val mdp = BlokusMDP(PlayerColor.RED, ExploitingMLN(network))

    uiServer.attach(statsStorage)
    network.listeners.add(StatsListener(statsStorage))

    mdp.actionOnReset = { blokusState ->
        println(blokusState.gameData.placementCounts.winningColor!!.toString() + " has won.")
        print(BoardUtil.visualizeBoard(blokusState.gameData.board))
    }

    val dql = QLearningSelectiveDiscreteDense(mdp, DQN(network), blokusQLConfig, manager)

    dql.train()
    mdp.close()
}
