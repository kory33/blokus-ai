package com.github.kory33.blokus.test

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.rl4j.util.Constants
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

private val LEARNING_RATE = 0.01

private fun getBlokusMLConfiguration(shape : IntArray, outputSize : Int) : MultiLayerConfiguration {
    val confB = NeuralNetConfiguration.Builder()
            .seed(Constants.NEURAL_NET_SEED)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(LEARNING_RATE)
            .updater(Updater.ADAM)
            .weightInit(WeightInit.XAVIER)
            .biasInit(1.0)
            .regularization(true)
            .list()

    confB.inputType = InputType.convolutionalFlat(shape[1], shape[2], shape[0])

    val layers = arrayListOf<Layer>(
            ConvolutionLayer.Builder(5, 5)
                    .nOut(100)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build(),
            ConvolutionLayer.Builder(4, 4)
                    .nOut(35)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build(),
            ConvolutionLayer.Builder(3, 3)
                    .nOut(10)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build(),
            DenseLayer.Builder()
                    .nOut(144)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build(),
            DenseLayer.Builder()
                    .nOut(144)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build(),
            OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.SOFTMAX)
                    .nOut(outputSize)
                    .build()
    )

    layers.forEachIndexed { index, layer -> confB.layer(index, layer) }

    return confB
            .pretrain(false)
            .backprop(true)
            .build()
}

fun getBlokusMLN(shape : IntArray, outputSize : Int) : MultiLayerNetwork {
    val model = MultiLayerNetwork(getBlokusMLConfiguration(shape, outputSize))
    model.init()
    return model
}