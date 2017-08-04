package com.github.kory33.blokus.test.util

import com.github.kory33.blokus.game.color.CellColor
import com.github.kory33.blokus.game.data.BlokusBoard
import com.github.kory33.blokus.util.IntegerVector


object BoardUtil {
    fun visualizeBoard(board: BlokusBoard): String {
        val result = StringBuilder()
        (1..12).forEach { row ->
            (1..12).forEach { column ->
                val cellCoordinate = IntegerVector(column, row)
                val cellColor = board.getCellColorAt(cellCoordinate)

                when (cellColor) {
                    CellColor.RED ->  result.append("\u001B[31mR ")
                    CellColor.BLUE -> result.append("\u001B[34mB ")
                    CellColor.NONE -> result.append("\u001B[0m. ")
                }
            }
            result.append("\n\u001B[0m")
        }
        return result.toString()
    }
}