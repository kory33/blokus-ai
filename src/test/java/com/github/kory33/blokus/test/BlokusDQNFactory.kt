package com.github.kory33.blokus.test

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.rl4j.util.Constants
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

private val LEARNING_RATE = 0.01
private val HIDDEN_NODES = 200
private val LAYERS = 5

private fun getBlokusMLConfiguration(shape : IntArray, outputSize : Int) : MultiLayerConfiguration {
    val confB = NeuralNetConfiguration.Builder()
            .seed(Constants.NEURAL_NET_SEED)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(LEARNING_RATE)
            //.updater(Updater.NESTEROVS).momentum(0.9)
            .updater(Updater.ADAM)
            //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
            .weightInit(WeightInit.XAVIER)
            .biasInit(1.0)
            //.regularization(true)
            //.l2(conf.getL2())
            .list()
            .layer(0, DenseLayer.Builder()
                    .nIn(shape[0])
                    .nOut(HIDDEN_NODES)
                    .activation(Activation.RELU)
                    .build())


    for (i in 1..LAYERS - 1) {
        confB.layer(i, DenseLayer.Builder()
                        .nIn(HIDDEN_NODES)
                        .nOut(HIDDEN_NODES)
                        .activation(Activation.RELU)
                        .build())
    }

    confB.layer(LAYERS, OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(HIDDEN_NODES)
                    .nOut(outputSize)
                    .build())


    return confB.pretrain(false).backprop(true).build()
}

fun getBlokusMLN(shape : IntArray, outputSize : Int) : MultiLayerNetwork {
    val model = MultiLayerNetwork(getBlokusMLConfiguration(shape, outputSize))
    model.init()
    return model
}