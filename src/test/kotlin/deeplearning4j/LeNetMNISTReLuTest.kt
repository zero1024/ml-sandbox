package deeplearning4j

import deeplearning4j.DataUtilities.downloadFile
import deeplearning4j.DataUtilities.extractTarGz
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator


/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.junit.jupiter.api.Test
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.slf4j.LoggerFactory
import java.io.File
import java.lang.Exception
import java.util.*
import kotlin.collections.HashMap


/**
 * Implementation of LeNet-5 for handwritten digits image classification on MNIST dataset (99% accuracy)
 * [[LeCun et al., 1998. Gradient based learning applied to document recognition]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
 * Some minor changes are made to the architecture like using ReLU and identity activation instead of
 * sigmoid/tanh, max pooling instead of avg pooling and softmax output layer.
 *
 *
 * This example will download 15 Mb of data on the first run.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 * @author dariuszzbyrad
 */
class LeNetMNISTReLu {
    private val LOGGER = LoggerFactory.getLogger(LeNetMNISTReLu::class.java)
    private val BASE_PATH = System.getProperty("user.home") + "/Downloads/dl4j-examples-data/mnist"
    private val DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"

    @Test
    fun test() {
        val height = 28L // height of the picture in px
        val width = 28L // width of the picture in px
        val channels = 1L // single channel for grayscale images
        val outputNum = 10 // 10 digits classification
        val batchSize = 54 // number of samples that will be propagated through the network in each iteration
        val nEpochs = 1 // number of training epochs
        val seed = 1234 // number used to initialize a pseudorandom number generator.
        val randNumGen = Random(seed.toLong())
        LOGGER.info("Data load...")
        if (!File(BASE_PATH + "/mnist_png").exists()) {
            LOGGER.debug("Data downloaded from {}", DATA_URL)
            val localFilePath = BASE_PATH + "/mnist_png.tar.gz"
            if (downloadFile(DATA_URL, localFilePath)) {
                extractTarGz(localFilePath, BASE_PATH)
            }
        }
        LOGGER.info("Data vectorization...")
        // vectorization of train data
        val trainData = File(BASE_PATH + "/mnist_png/training")
        val trainSplit = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val labelMaker = ParentPathLabelGenerator() // use parent directory name as the image label
        val trainRR = ImageRecordReader(height, width, channels, labelMaker)
        trainRR.initialize(trainSplit)
        val trainIter: DataSetIterator = RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum)

        // pixel values from 0-255 to 0-1 (min-max scaling)
        val imageScaler: DataNormalization = ImagePreProcessingScaler()
        imageScaler.fit(trainIter)
        trainIter.preProcessor = imageScaler

        // vectorization of test data
        val testData = File(BASE_PATH + "/mnist_png/testing")
        val testSplit = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val testRR = ImageRecordReader(height, width, channels, labelMaker)
        testRR.initialize(testSplit)
        val testIter: DataSetIterator = RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum)
        testIter.preProcessor = imageScaler // same normalization for better results
        LOGGER.info("Network configuration and training...")
        // reduce the learning rate as the number of training epochs increases
        // iteration #, learning rate
        val learningRateSchedule: MutableMap<Int, Double> = HashMap()
        learningRateSchedule[0] = 0.06
        learningRateSchedule[200] = 0.05
        learningRateSchedule[600] = 0.028
        learningRateSchedule[800] = 0.0060
        learningRateSchedule[1000] = 0.001
        val conf = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(0.0005) // ridge regression value
            .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(
                ConvolutionLayer.Builder(5, 5)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .setInputType(
                InputType.convolutionalFlat(
                    height.toLong(),
                    width.toLong(),
                    channels.toLong()
                )
            ) // InputType.convolutional for normal image
            .build()
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(10))
        LOGGER.info("Total num of params: {}", net.numParams())

        // evaluation while training (the score should go down)
        for (i in 0 until nEpochs) {
            net.fit(trainIter)
            LOGGER.info("Completed epoch {}", i)
            val eval = net.evaluate<Evaluation>(testIter)
            LOGGER.info(eval.stats())
            trainIter.reset()
            testIter.reset()
        }
        val ministModelPath = File(BASE_PATH + "/minist-model.zip")
        ModelSerializer.writeModel(net, ministModelPath, true)
        LOGGER.info("The MINIST model has been saved in {}", ministModelPath.path)
    }
}