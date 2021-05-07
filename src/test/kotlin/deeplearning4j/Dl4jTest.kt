package deeplearning4j

import deeplearning4j.DownloaderUtility
import deeplearning4j.PlotUtil
import org.junit.jupiter.api.Test
import java.util.concurrent.TimeUnit
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.learning.config.Nesterovs
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.datavec.api.split.FileSplit
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.records.reader.RecordReader
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.lang.Exception


class Dl4jTest {

    @Test
    internal fun test() {
        val batchSize = 100
        val seed = 123
        val learningRate = 0.005
        //Number of epochs (full passes of the data)
        val nEpochs = 30
        val numInputs = 2
        val numOutputs = 2
        val numHiddenNodes = 20
        dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download()
        //Load the training data:
        val rr: RecordReader = CSVRecordReader()
        rr.initialize(FileSplit(File(dataLocalPath, "saturn_data_train.csv")))
        val trainIter: DataSetIterator = RecordReaderDataSetIterator(rr, batchSize, 0, 2)

        //Load the test/evaluation data:
        val rrTest: RecordReader = CSVRecordReader()
        rrTest.initialize(FileSplit(File(dataLocalPath, "saturn_data_eval.csv")))
        val testIter: DataSetIterator = RecordReaderDataSetIterator(rrTest, batchSize, 0, 2)

        //log.info("Build model....");
        val conf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .weightInit(WeightInit.XAVIER)
            .updater(Nesterovs(learningRate, 0.9))
            .list()
            .layer(
                DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX)
                    .nIn(numHiddenNodes).nOut(numOutputs).build()
            )
            .build()
        val model = MultiLayerNetwork(conf)
        model.init()
        model.setListeners(ScoreIterationListener(10)) //Print score every 10 parameter updates
        model.fit(trainIter, nEpochs)
        println("Evaluate model....")
        val eval: Evaluation = model.evaluate(testIter)
        System.out.println(eval.stats())
        println("\n****************Example finished********************")

        //Training is complete. Code that follows is for plotting the data & predictions only
        generateVisuals(model, trainIter, testIter)
        Thread.sleep(1000000)
    }

    var dataLocalPath: String? = null
    var visualize = true


    @Throws(Exception::class)
    fun generateVisuals(model: MultiLayerNetwork?, trainIter: DataSetIterator?, testIter: DataSetIterator?) {
        if (visualize) {
            val xMin = -15.0
            val xMax = 15.0
            val yMin = -15.0
            val yMax = 15.0

            //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
            val nPointsPerAxis = 100

            //Generate x,y points that span the whole range of features
            val allXYPoints: INDArray = PlotUtil.generatePointsOnGraph(xMin, xMax, yMin, yMax, nPointsPerAxis)
            //Get train data and plot with predictions
            PlotUtil.plotTrainingData(model, trainIter, allXYPoints, nPointsPerAxis)
            TimeUnit.SECONDS.sleep(3)
            //Get test data, run the test data through the network to generate predictions, and plot those predictions:
            PlotUtil.plotTestData(model, testIter, allXYPoints, nPointsPerAxis)
        }
    }

}