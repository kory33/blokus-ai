package com.github.kory33.blokus.environment

import com.github.kory33.blokus.game.BlokusPlacement
import com.github.kory33.blokus.game.BlokusPlacementSpace
import com.github.kory33.blokus.game.ColoredBlokusPlacement
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BlokusActionSpace(possibleActions: Set<ColoredBlokusPlacement>)
        : SelectiveDiscreteSpace(size) {

    val actionAvailabilityArray: DoubleArray = DoubleArray(BlokusPlacementSpace.PLACEMENT_LIST.size)
    var availableActionCache : List<Int>
    init {
        val actionList = ArrayList<Int>()

        possibleActions
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

    override fun maxQValue(qVector: INDArray): Double {
        return qVector.getDouble(maxQValueAction(qVector))
    }

    override fun maxQValueAction(qVector: INDArray): Int {
        var maxQValue : Double? = null
        var maxQValueIndex : Int? = null
        getAvailableActions().forEach { index ->
            val q = qVector.getDouble(index)
            if (maxQValue == null || q > maxQValue!!) {
                maxQValue = q
                maxQValueIndex = index
            }
        }

        return maxQValueIndex!!
    }

    override fun filterQValues(qVector: INDArray): INDArray {
        return qVector.muliRowVector(computeActionAvailability())
    }

    companion object {
        fun getPlacementCorrespondingToIndex(index : Int) : BlokusPlacement? {
            return BlokusPlacementSpace.PLACEMENT_LIST[index]
        }
        val size = BlokusPlacementSpace.PLACEMENT_LIST.size
    }
}