import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
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
        SparkSession sparkSession = Utility.InitializeSparkSession();
        Dataset<Row> wineDataFrame = Utility.ReadDataframeFromCsvFile(sparkSession, "TrainingDataset.csv");

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(Utility.FEATURE_COLUMNS)
                .setOutputCol("features");
        Dataset<Row> assemblyResult = vectorAssembler.transform(wineDataFrame).select("quality", "features");

        LogisticRegression logisticRegression = new LogisticRegression()
                .setFeaturesCol("features")
                .setRegParam(0.2)
                .setMaxIter(15)
                .setLabelCol("quality");
        LogisticRegressionModel lrModel = logisticRegression.fit(assemblyResult);
        lrModel.write().overwrite().save("model");
    }

}
