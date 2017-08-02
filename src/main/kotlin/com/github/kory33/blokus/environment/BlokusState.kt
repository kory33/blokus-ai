package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusGame
import com.github.kory33.blokus.game.BlokusPlacement
import com.github.kory33.blokus.game.ColoredBlokusPlacement
import com.github.kory33.blokus.game.IBlokusPlayer
import com.github.kory33.blokus.game.color.PlayerColor
import com.github.kory33.blokus.game.data.BlokusGameData
import org.deeplearning4j.gym.StepReply
import org.deeplearning4j.rl4j.space.Encodable
import org.json.JSONObject

class BlokusState(private val exploitingAdversary: IBlokusPlayer,
                  private val playerColor: PlayerColor) : Encodable{
    private val blokusGame = BlokusGame()

    init {
        exploitingAdversary.assignColor(playerColor.opponentColor)
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

        val adversaryPlacement =
                exploitingAdversary.chooseBestPlacementFrom(blokusGame.possiblePlacements, gameData)
        blokusGame.makePlacement(adversaryPlacement)
    }

    fun hasGameFinished() : Boolean = blokusGame.isGameFinished

    val gameData : BlokusGameData
        get() = blokusGame.gameData

    val information : JSONObject = JSONObject("{}")

    private fun getReward() : Double {
        if (!this.hasGameFinished()) {
            return 0.0
        }

        val placementCounts = gameData.placementCounts.toMap()
        val playerPlacementSize = placementCounts[playerColor]!!
        val adversaryPlacementSize = placementCounts[playerColor.opponentColor]!!

        val winningBias = if (this.blokusGame.winnerColor!! == playerColor) 10.0 else - 10.0

        return playerPlacementSize - adversaryPlacementSize + winningBias
    }

    val observation : BlokusObservation
        get() = BlokusObservation(this, this.playerColor)

    private val reply
        get() = StepReply(this, this.getReward(), this.hasGameFinished(), this.information)

    fun step(action : BlokusPlacement) : StepReply<BlokusState> {
        blokusGame.makePlacement(findMatchingAction(action)!!)

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