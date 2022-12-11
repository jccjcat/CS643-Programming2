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

    public static SparkSession InitializeSparkSession() {
        return SparkSession.builder()
                .appName(APP_NAME)
                .master("local[" + THREADS_TO_USE + "]")
                .getOrCreate();
    }

    public static Dataset<Row> ReadDataframeFromCsvFile(SparkSession sparkSession, String path) {
        return sparkSession.read()
                .option("header", true)
                .option("delimiter", ";")
                .option("escape", "\"")
                .option("inferSchema", true)
                .csv(path);
    }

}
