package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusPlacement
import com.github.kory33.blokus.game.BlokusPlacementSpace
import com.github.kory33.blokus.util.BijectiveHashMap
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BlokusActionSpace : SelectiveDiscreteSpace<BlokusState>(BlokusPlacementSpace.PLACEMENT_LIST.size) {
    val actionMap : BijectiveHashMap<Int, BlokusPlacement> = BijectiveHashMap()

    init {
        BlokusPlacementSpace.PLACEMENT_LIST.forEachIndexed { index, blokusPlacement ->
            actionMap.put(index, blokusPlacement)
        }
    }

    /**
     * Computes a vector with elements 1 if the action corresponding to the
     * index of the element is valid, 0 if not.
     */
    override fun computeActionAvailability(state: BlokusState) : INDArray {
        val availabilityList = BlokusPlacementSpace.PLACEMENT_LIST
                .map { state.findMatchingAction(it) != null }
                .map { isValidAction -> if (isValidAction) 1.0 else 0.0 }

        return Nd4j.create(availabilityList.toDoubleArray())
    }

    fun getPlacementCorrespondingToIndex(index : Int) : BlokusPlacement? {
        return actionMap.getValue(index)
    }

    fun getIndexOf(placement : BlokusPlacement) : Int? {
        return actionMap.getKey(placement)
    }
}