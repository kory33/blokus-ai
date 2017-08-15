package com.github.kory33.blokus.ai

import org.apache.commons.collections4.queue.CircularFifoQueue
import org.deeplearning4j.rl4j.space.Encodable
import java.util.*

class ObservableExpReplay<A, O : Encodable>(private val batchSize: Int,
                                            maxQueueSize: Int): IObservableExpReplay<A, O> {

    private val storage = CircularFifoQueue<ObservableTransition<A, O>>(maxQueueSize)

    fun getBatch(size: Int): ArrayList<ObservableTransition<A, O>> {
        val random = Random()
        val intSet = HashSet<Int>()
        val storageSize = storage.size
        while (intSet.size < size) {
            val rd = random.nextInt(storageSize)
            intSet.add(rd)
        }

        val batch = ArrayList<ObservableTransition<A, O>>(size)
        val iter = intSet.iterator()
        while (iter.hasNext()) {
            val trans = storage.get(iter.next())
            batch.add(trans)
        }

        return batch
    }

    override fun getBatch() = this.getBatch(batchSize)

    override fun store(transition: ObservableTransition<A, O>) {
        storage.add(transition)
    }
}