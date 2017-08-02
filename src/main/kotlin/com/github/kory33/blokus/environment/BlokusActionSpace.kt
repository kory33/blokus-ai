package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusPlacement
import com.github.kory33.blokus.game.BlokusPlacementSpace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BlokusActionSpace(private val state: BlokusState)
        : SelectiveDiscreteSpace(BlokusPlacementSpace.PLACEMENT_LIST.size) {

    val actionAvailabilityArray: DoubleArray
    var availableActions : List<Int>
    init {
        actionAvailabilityArray = BlokusPlacementSpace.PLACEMENT_LIST
                .map { placement ->
                    state.possibleActions.firstOrNull { placement.cellCoordinates == it.cellCoordinates } != null
                }
                .map { isValidAction -> if (isValidAction) 1.0 else 0.0 }
                .toDoubleArray()

        val actionList = ArrayList<Int>()
        (0 .. actionAvailabilityArray.size - 1).forEach {
            if (actionAvailabilityArray[it] != 0.0) actionList.add(it)
        }
        availableActions = actionList
    }

    override fun computeActionAvailability(): INDArray = Nd4j.create(actionAvailabilityArray)
    override fun randomAction(): Int = availableActions[rd.nextInt(availableActions.size)]

    companion object {
        fun getPlacementCorrespondingToIndex(index : Int) : BlokusPlacement? {
            return BlokusPlacementSpace.PLACEMENT_LIST[index]
        }
    }
}