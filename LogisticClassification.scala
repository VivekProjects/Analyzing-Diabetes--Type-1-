import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object LogisticClassification extends App {

  Logger.getLogger("org").setLevel(Level.OFF)
  val spark = SparkSession.builder
    .master("local[*]")
    .appName("ZSpark23")
    .getOrCreate()

  import spark.implicits._
  spark.version
  val fileName = "C:\\Users\\wisem\\Downloads\\diabetes.csv"


  val dataDF  = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(fileName).toDF().withColumnRenamed("Outcome", "label")
  print("Original Data that is Loaded to Spark!!!!!!!!!!!!!!!!!!!!!!!!!!")
  dataDF.show()

  val assembler = new VectorAssembler()
    .setInputCols(Array("Glucose") )
    .setOutputCol("features")


  val readyDF = assembler.transform(dataDF)
  print("After transformation through Vector!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  readyDF.show()
  val cleandata = readyDF.filter(arr => arr(0) != "")
  cleandata.count()
  cleandata.describe().select("Summary","Glucose").show()
  //cleandata.describe().select("Summary","SkinThickness","Insulin").show()
  //cleandata.describe().select("Summary","BMI","DiabetesPedigreeFunction","Age").show()

  val newdata = cleandata.randomSplit(Array(.70,.30), seed = 11L)
print("After the data is split using random split!!!!!!!!!!!!!!!!!!!!!!")
  val trainingdataset = newdata(0)
  val testingdataset = newdata(1)
  println("Number of training data:"+ trainingdataset.count() )
  println("Number of testing data:"+ testingdataset.count() )

  val  lr = new LogisticRegression().setMaxIter(6)
  val lrModel = lr.fit(trainingdataset)
  println(s"Coefficients = ${lrModel.coefficients}, Intercept = ${lrModel.intercept}")

  val predictions = lrModel.transform(testingdataset)
  predictions.show(100)

  val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
  val accuracy = evaluator.evaluate(predictions)
  println(s"Accuracy_AUC: ${accuracy}")

  val lp = predictions.select( "label", "prediction")
  val counttotal = predictions.count()
  println(s"Count_total: ${counttotal}")
  val correct = lp.filter($"label" === $"prediction").count()
  println(s"Correct: ${correct}")
  val wrong = lp.filter(($"label" =!= $"prediction")).count()
  println(s"Wrong: ${wrong}")
  val truepositive = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count()
  println(s"Truepositive: ${truepositive}")
  val truenegative = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
  println(s"Truenegative: ${truenegative}")
  val falseNegative = lp.filter($"prediction" === 0.0).filter(($"label" =!= $"prediction")).count()
  println(s"FalseNegative: ${falseNegative}")
  val falsePositive = lp.filter($"prediction" === 1.0).filter(($"label" =!= $"prediction")).count()
  println(s"FalsePositive: ${falsePositive}")
  val ratioWrong= wrong.toDouble/counttotal.toDouble
  println(s"RatioWrong: ${ratioWrong}")
  val ratioCorrect = correct.toDouble/counttotal.toDouble
  println(s"RatioCorrect: ${ratioCorrect}")

}