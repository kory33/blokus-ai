package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusConstant
import com.github.kory33.blokus.game.color.CellColor
import com.github.kory33.blokus.game.color.PlayerColor
import com.github.kory33.blokus.game.data.BlokusBoard
import com.github.kory33.blokus.util.IntegerVector
import org.deeplearning4j.rl4j.space.Encodable

/**
 * Represents an observation of a state
 */
class BlokusObservation(state: BlokusState, private val playerColor: PlayerColor) : Encodable {
    private val board : BlokusBoard = state.gameData.board
    private var arrayRepresentation : DoubleArray? = null
    private var swappedArrayRepresentation : DoubleArray? = null

    private fun getArrayListOfColor(validColor : CellColor) : DoubleArray {
        val resultList = ArrayList<Double>()

        val size = BlokusConstant.BOARD_SIZE
        (1..size).forEach({ row ->
            (1..size).forEach({ column ->
                val vector = IntegerVector(column, row)
                val cellValue = if (this.board.getCellColorAt(vector) == validColor) 1.0 else 0.0
                resultList.add(cellValue)
            })
        })

        return resultList.toDoubleArray()
    }

    private fun updateCache() {
        val playerColorBoardArray = getArrayListOfColor(CellColor.fromPlayerColor(this.playerColor))
        val adversaryColorBoardArray = getArrayListOfColor(CellColor.fromPlayerColor(this.playerColor.opponentColor))

        this.arrayRepresentation = playerColorBoardArray + adversaryColorBoardArray
        this.swappedArrayRepresentation = adversaryColorBoardArray + playerColorBoardArray
    }

    fun toSwappedArray() : DoubleArray {
        if (this.swappedArrayRepresentation != null) {
            return this.swappedArrayRepresentation!!
        }

        updateCache()
        return this.swappedArrayRepresentation!!
    }

    /**
     * Get an array-representation of this observation.
     *
     * In the returned array, first 144 elements consists of 0.0 or 1.0,
     * high if the corresponding cell is marked as the player's color.
     * Latter 144 elements are similarly computed but with the color of the adversary.
     */
    override fun toArray(): DoubleArray {
        if (this.arrayRepresentation != null) {
            return this.arrayRepresentation!!
        }

        updateCache()
        return this.arrayRepresentation!!
    }
}
