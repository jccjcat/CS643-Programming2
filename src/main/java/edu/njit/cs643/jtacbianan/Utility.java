package edu.njit.cs643.jtacbianan;

import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
/**
 * Application Shared Utility Class
 * @author Jan Chris Tacbianan
 */
public class Utility {

    //Constants
    public static final String APP_NAME = "CS643-Assignment2";
    public static final int THREADS_TO_USE = 4;
    public static final String[] FEATURE_COLUMNS =  {
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
    };

    public static SparkSession initializeSparkSession() {
        SparkSession session =  SparkSession.builder()
                .appName(APP_NAME)
                .master("local[" + THREADS_TO_USE + "]")
                .getOrCreate();
        session.sparkContext().setLogLevel("OFF");
        return session;
    }

    public static Dataset<Row> readDataframeFromCsvFile(SparkSession sparkSession, String path) {
        return sparkSession.read()
                .option("header", true)
                .option("delimiter", ";")
                .option("escape", "\"")
                .option("inferSchema", true)
                .csv(path);
    }

    public static Dataset<Row> assembleDataframe(Dataset<Row> dataframe) {
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(Utility.FEATURE_COLUMNS)
                .setOutputCol("features");
        Dataset<Row> assemblerResult = vectorAssembler.transform(dataframe);
        return assemblerResult.select("quality", "features");

    }

    public static void evaluateAndSummarizeDataModel(Dataset<Row> dataFrame) {
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction");

        double accuracy = evaluator.setMetricName("accuracy").evaluate(dataFrame);
        double f1 = evaluator.setMetricName("f1").evaluate(dataFrame);
        System.out.println("Model Accuracy:  " + accuracy);
        System.out.println("F1 Score: " + f1);
    }

    public static Dataset<Row> transformDataframeWithModel(LogisticRegressionModel model, Dataset<Row> dataFrame) {
        return model.transform(dataFrame).select("features", "quality", "prediction");
    }

}
