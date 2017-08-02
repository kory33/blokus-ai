package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusGame
import com.github.kory33.blokus.game.BlokusPlacement
import com.github.kory33.blokus.game.ColoredBlokusPlacement
import com.github.kory33.blokus.game.IBlokusPlayer
import com.github.kory33.blokus.game.color.PlayerColor
import com.github.kory33.blokus.game.data.BlokusGameData
import org.deeplearning4j.gym.StepReply
import org.json.JSONObject

class BlokusState(private val exploitingAdversary: IBlokusPlayer, private val playerColor: PlayerColor) {
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
    val possibleActions : Set<ColoredBlokusPlacement> = blokusGame.possiblePlacements.toSet()

    fun findMatchingAction(placement : BlokusPlacement) : ColoredBlokusPlacement? =
            this.possibleActions.firstOrNull { placement.cellCoordinates == it.cellCoordinates }

    fun letAdversaryPlay() {
        if (this.hasGameFinished()) {
            return
        }

        val adversaryPlacement =
                exploitingAdversary.chooseBestPlacementFrom(blokusGame.possiblePlacements, this.blokusGame.gameData)
        blokusGame.makePlacement(adversaryPlacement)
    }

    fun hasGameFinished() : Boolean = blokusGame.isGameFinished

    val gameData : BlokusGameData = blokusGame.gameData

    val information : JSONObject = JSONObject("{}")

    private fun getReward() : Double {
        if (!this.hasGameFinished()) {
            return 0.0
        }

        val placementCounts = this.blokusGame.gameData.placementCounts.toMap()
        val playerPlacementSize = placementCounts[playerColor]!!
        val adversaryPlacementSize = placementCounts[playerColor.opponentColor]!!

        val winningBias = if (this.blokusGame.winnerColor!! == playerColor) 10.0 else - 10.0

        return playerPlacementSize - adversaryPlacementSize + winningBias
    }

    val observation : BlokusObservation = BlokusObservation(this, this.playerColor)

    private val reply = StepReply(this.observation, this.getReward(), this.hasGameFinished(), this.information)

    fun step(action : BlokusPlacement) : StepReply<BlokusObservation> {
        blokusGame.makePlacement(findMatchingAction(action)!!)

        if (blokusGame.phase.nextPlayerColor == playerColor) {
            return this.reply
        }

        if (blokusGame.phase.nextPlayerColor == playerColor.opponentColor) {
            letAdversaryPlay()
            if (blokusGame.phase.nextPlayerColor == playerColor.opponentColor) {
                while(!this.hasGameFinished()) {
                    letAdversaryPlay()
                }
            }
        }

        return this.reply
    }
}