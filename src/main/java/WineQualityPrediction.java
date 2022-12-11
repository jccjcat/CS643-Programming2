import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
/**
 * Wine Quality Prediction Application
 * @author Jan Chris Tacbianan
 */
public class WineQualityPrediction {
    /**
     * Main Method
     * - Application Entry Point
     * @param args Command Line Arguments
     */
    public static void main(String[] args) {
        System.out.println("Initializng Spark:");
        SparkSession sparkSession = Utility.initializeSparkSession();

        System.out.println("Loading Test Data Set");

        //Load Test Dataframe
        Dataset<Row> testingDataFrame = Utility.readDataframeFromCsvFile(sparkSession, "TestDataset.csv");
        Dataset<Row> assembledTestDataFrame = Utility.assembleDataframe(testingDataFrame);

        System.out.println("Loading Training Model");

        //Load Training Model
        LogisticRegressionModel lrModel = LogisticRegressionModel.load("model");

        System.out.println("Predicting using Trained Model and Test Data");

        //Predict using Test Dataframe
        Dataset<Row> predictionData = Utility.transformDataframeWithModel(lrModel, assembledTestDataFrame);

        predictionData.show();

        System.out.println("Evaluation Results:");

        Utility.evaluateAndSummarizeDataModel(predictionData);
    }
}
