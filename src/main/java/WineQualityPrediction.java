import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
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
        SparkSession sparkSession = Utility.InitializeSparkSession();
        Dataset<Row> validationDataFrame = Utility.ReadDataframeFromCsvFile(sparkSession, "ValidationDataset.csv");
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(Utility.FEATURE_COLUMNS)
                .setOutputCol("features");
        Dataset<Row> assemblerResult = vectorAssembler.transform(validationDataFrame).select("quality", "features");

        LogisticRegressionModel lrModel = LogisticRegressionModel.load("model");

        Dataset<Row> predictionData = lrModel.transform(assemblerResult).select("features", "quality", "prediction");

        predictionData.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction");

        double accuracy = evaluator.setMetricName("accuracy").evaluate(predictionData);
        double f1 = evaluator.setMetricName("f1").evaluate(predictionData);
        System.out.println("Model Accuracy:  " + accuracy);
        System.out.println("F1 Score: " + f1);
    }
}
