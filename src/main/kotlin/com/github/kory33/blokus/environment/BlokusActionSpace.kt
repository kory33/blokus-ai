package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusPlacement
import com.github.kory33.blokus.game.BlokusPlacementSpace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BlokusActionSpace(private val state: BlokusState)
        : SelectiveDiscreteSpace(BlokusPlacementSpace.PLACEMENT_LIST.size) {

    val actionAvailabilityArray: DoubleArray = BlokusPlacementSpace.PLACEMENT_LIST
                .map { state.findMatchingAction(it) != null }
                .map { isValidAction -> if (isValidAction) 1.0 else 0.0 }
                .toDoubleArray()

    var availableActions : List<Int>
    init {
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