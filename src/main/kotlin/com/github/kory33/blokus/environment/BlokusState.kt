package com.github.kory33.blokus.environment

import com.github.kory33.blokus.ai.ExploitingMLN
import com.github.kory33.blokus.environment.space.BlokusActionSpace
import com.github.kory33.blokus.game.BlokusGame
import com.github.kory33.blokus.game.BlokusPlacement
import com.github.kory33.blokus.game.ColoredBlokusPlacement
import com.github.kory33.blokus.game.color.PlayerColor
import com.github.kory33.blokus.game.data.BlokusGameData
import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.space.Encodable
import org.json.JSONObject
import org.nd4j.linalg.factory.Nd4j

class BlokusState(private val exploitingMLN : ExploitingMLN<BlokusActionSpace>,
                  private val playerColor: PlayerColor) : Encodable{
    private val blokusGame = BlokusGame()
    private var rewardReservoir = 0

    fun getActionSpace() : BlokusActionSpace = BlokusActionSpace(blokusGame.possiblePlacements)

    init {
        if (playerColor == PlayerColor.BLUE) {
            letAdversaryPlay()
        }
    }

    /**
     * Get A(s) where s = this
     */
    val possibleActions : Set<ColoredBlokusPlacement>
        get() = blokusGame.possiblePlacements.toSet()

    fun findMatchingAction(placement : BlokusPlacement) : ColoredBlokusPlacement? =
            this.possibleActions.firstOrNull { placement.cellCoordinates == it.cellCoordinates }

    fun letAdversaryPlay() {
        if (this.hasGameFinished()) {
            return
        }

        val observation = Nd4j.create(this.observation.toReversedArray())
        val adversaryOutput = exploitingMLN.getNextAction(observation, getActionSpace())
        val adversaryAction = BlokusActionSpace.getPlacementCorrespondingToIndex(adversaryOutput)!!
        val gameAction = findMatchingAction(adversaryAction)!!

        rewardReservoir -= gameAction.size()
        blokusGame.makePlacement(gameAction)
    }

    fun hasGameFinished() : Boolean = blokusGame.isGameFinished

    val gameData : BlokusGameData
        get() = blokusGame.gameData

    val information : JSONObject = JSONObject("{}")

    private fun getReward() : Double {
        val winningBias = when (this.blokusGame.winnerColor) {
            playerColor -> 10.0
            playerColor.opponentColor -> -10.0
            else -> 0.0
        }

        val reward = rewardReservoir + winningBias
        rewardReservoir = 0
        return reward
    }

    val observation : BlokusObservation
        get() = BlokusObservation(this, this.playerColor)

    private val reply
        get() = StepReply(this, this.getReward(), this.hasGameFinished(), this.information)

    fun step(action : BlokusPlacement) : StepReply<BlokusState> {
        val gameAction = findMatchingAction(action)!!

        rewardReservoir += gameAction.size()
        blokusGame.makePlacement(gameAction)

        if (blokusGame.phase.nextPlayerColor == playerColor) {
            return reply
        }

        if (blokusGame.phase.nextPlayerColor == playerColor.opponentColor) {
            letAdversaryPlay()
            if (blokusGame.phase.nextPlayerColor == playerColor.opponentColor) {
                while(!this.hasGameFinished()) {
                    letAdversaryPlay()
                }
            }
        }

        return reply
    }

    override fun toArray() = this.observation.toArray()
}