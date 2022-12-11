package edu.njit.cs643.jtacbianan;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

/**
 * Model Training Application
 * @author Jan Chris Tacbianan
 */
public class ModelTraining {

    /**
     * Main Method
     * - Application Entry Point
     * @param args Command Line Arguments
     */
    public static void main(String[] args) throws IOException {
        System.out.println("Model Training Application");

        System.out.println("Initializng Spark:");

        SparkSession sparkSession = Utility.initializeSparkSession();

        System.out.println("Training Logistic Regression Model...");

        //Initial Training
        Dataset<Row> wineDataFrame = Utility.readDataframeFromCsvFile(sparkSession, "TrainingDataset.csv");
        Dataset<Row> assemblyResult = Utility.assembleDataframe(wineDataFrame);

        LogisticRegression logisticRegression = new LogisticRegression()
                .setFeaturesCol("features")
                .setRegParam(0.2)
                .setMaxIter(15)
                .setLabelCol("quality");
        LogisticRegressionModel lrModel = logisticRegression.fit(assemblyResult);

        System.out.println("Validating Trained Model");

        //Validating Trained Model
        Dataset<Row> validationDataFrame = Utility.readDataframeFromCsvFile(sparkSession, "ValidationDataset.csv");
        Dataset<Row> assembledValidationDataFrame = Utility.assembleDataframe(validationDataFrame);
        Dataset<Row> modelTransformationResult = Utility.transformDataframeWithModel(lrModel, assembledValidationDataFrame);

        System.out.println("Validation Results");

        //Print Results
        Utility.evaluateAndSummarizeDataModel(modelTransformationResult);

        System.out.println("Saving trained model.");

        //Save new model.
        lrModel.write().overwrite().save("model");

        sparkSession.stop();
    }

}
