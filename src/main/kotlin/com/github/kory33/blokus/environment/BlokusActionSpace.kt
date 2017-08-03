package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusPlacement
import com.github.kory33.blokus.game.BlokusPlacementSpace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BlokusActionSpace(state: BlokusState)
        : SelectiveDiscreteSpace(BlokusPlacementSpace.PLACEMENT_LIST.size) {

    val actionAvailabilityArray: DoubleArray
    var availableActionCache : List<Int>
    init {
        actionAvailabilityArray = kotlin.DoubleArray(BlokusPlacementSpace.PLACEMENT_LIST.size)
        val actionList = ArrayList<Int>()

        state.possibleActions
                .map { placement ->
                    BlokusPlacementSpace.PLACEMENT_INDEX_MAP.getKey(placement)!!
                }
                .forEach {index ->
                    actionAvailabilityArray[index] = 1.0
                    actionList.add(index)
                }
        availableActionCache = actionList
    }

    override fun computeActionAvailability(): INDArray = Nd4j.create(actionAvailabilityArray)
    override fun getAvailableActions(): List<Int> = availableActionCache

    companion object {
        fun getPlacementCorrespondingToIndex(index : Int) : BlokusPlacement? {
            return BlokusPlacementSpace.PLACEMENT_LIST[index]
        }
    }
}