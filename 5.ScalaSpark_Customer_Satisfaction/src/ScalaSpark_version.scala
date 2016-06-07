
import java.io.File

import framian.csv.Csv
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
//import org.saddle.{Frame, mat}


// run with VM arguments = -Dspark.master=local
// -------------------------------------------------------------------------------------------------------
// OK
// -------------------------------------------------------------------------------------------------------

val conf = new SparkConf().setAppName("ScalaSpark_version")
val sc = new SparkContext(conf)
val sqlContext = new org.apache.spark.sql.SQLContext(sc)

// -------------------------------------------------------------------------------------------------------
// OK
// -------------------------------------------------------------------------------------------------------

// Test
//val textFile = sc.textFile("README.md")
//println(textFile.count())

// -------------------------------------------------------------------------------------------------------
// OK
// -------------------------------------------------------------------------------------------------------

// LOAD DATA FILES (CSV)
val training_df = sqlContext.read.format("com.databricks.spark.csv")
  .option("inferSchema","true")
  .option("header","true")
  .load("5.ScalaSpark_Customer_Satisfaction/train.csv")

val test_df = sqlContext.read.format("com.databricks.spark.csv")
  .option("inferSchema","true")
  .option("header","true")
  .load("5.ScalaSpark_Customer_Satisfaction/test.csv")
//

// -------------------------------------------------------------------------------------------------------
// OK
// -------------------------------------------------------------------------------------------------------

// Example: show target
//training_df.select("TARGET").show(30) // (returns another dataframe and shows it)


// -------------------------------------------------------------------------------------------------------
// OK
// -------------------------------------------------------------------------------------------------------

// EXAMPLE: show age and target. add 1 unit to age. This is also an example of operations over columns

//training_df.select(training_df("var15"), training_df("var15")+1, training_df("TARGET")).show(50)

// The previous operation is based on DataFrame.select(Column*). The way to obtain a Column object from
// the training_df is using training_df("columnName"), although two other possibilities exist:
  // training_df.col("columnName")
  // training_df.apply("columnName")

// training_df("columnName") and training_df.col("columnName") seem more intuitive, but the three ways are correct.
// I haven't found a comparison of these three accessing methods yet.

//training_df.select(training_df.col("var15"), training_df.col("var15")+1, training_df.col("TARGET")).show(50)
//training_df.select(training_df.apply("var15"), training_df.apply("var15")+1, training_df.apply("TARGET")).show(50)


// SPEED BENCHMARK:
/*
var start = System.nanoTime // var, because it is not a constant
training_df.select(training_df("var15"), training_df("var15")+1, training_df("TARGET")).show(50)
val stop1 = System.nanoTime-start

start = System.nanoTime
training_df.select(training_df.col("var15"), training_df.col("var15")+1, training_df.col("TARGET")).show(50)
val stop2 = System.nanoTime-start

start = System.nanoTime
training_df.select(training_df.apply("var15"), training_df.apply("var15")+1, training_df.apply("TARGET")).show(50)
val stop3 = System.nanoTime-start

println("time df(\"columnName\"): "+(stop1)/1e6+"ms")
println("time df.col(\"columnName\"): "+(stop2)/1e6+"ms")
println("time df.apply(\"columnName\"): "+(stop3)/1e6+"ms")
*/

/* Results:
  time df("columnName"): 793.839325ms
  time df.col("columnName"): 297.516105ms
  time df.apply("columnName"): 171.635054ms

  seems that the less intuitive method, df.apply(), is the most efficient. Why? maybe it is just a matter of cache memory?
  let's change the order of execution:
*/

/*
var start = System.nanoTime
training_df.select(training_df.apply("var15"), training_df.apply("var15")+1, training_df.apply("TARGET")).show(50)
val stop3 = System.nanoTime-start

start = System.nanoTime
training_df.select(training_df.col("var15"), training_df.col("var15")+1, training_df.col("TARGET")).show(50)
val stop2 = System.nanoTime-start

start = System.nanoTime // var, because it is not a constant
training_df.select(training_df("var15"), training_df("var15")+1, training_df("TARGET")).show(50)
val stop1 = System.nanoTime-start


println("time df(\"columnName\"): "+(stop1)/1e6+"ms")
println("time df.col(\"columnName\"): "+(stop2)/1e6+"ms")
println("time df.apply(\"columnName\"): "+(stop3)/1e6+"ms")
*/

/* Results:
time df("columnName"): 146.263428ms
time df.col("columnName"): 145.227378ms
time df.apply("columnName"): 600.303015ms

Yes, it was just a matter of cache memory
*/


// example:
//println(training_df.select("var15", "TARGET").getClass)


// -------------------------------------------------------------------------------------------------------
// TO-DO
// -------------------------------------------------------------------------------------------------------
  // INTERESTING STUFF. I FOUND THE SCALA EQUIVALENTS TO PYTHON-PANDAS: Saddle, Scala-datatable and Framian.
// https://darrenjw.wordpress.com/2015/08/21/data-frames-and-tables-in-scala/

// Saddle: It can't even compile, seems buggy:
//val f: Frame[Int, Int, Double] = mat.rand(2, 2)

// Framian: to emulate pyspark training_df.describe().toPandas() method in scala, save training_df.describe()
// as a csv file and load it with Framian Csv.ParseFile:
//training_df.describe().write.format("com.databricks.spark.csv").save("5.ScalaSpark_Customer_Satisfaction/describe.csv")
//val df = Csv.parseFile(new File("5.ScalaSpark_Customer_Satisfaction/describe.csv")).labeled.toFrame

//println(df.rows)
// Error:
// java.lang.NoSuchMethodError: scala.Predef$.$conforms()Lscala/Predef$$less$colon$less;
// due to incompatibilities between Spark and Scala versions. Solution: download Spark sources and compile with the
// installed version of Scala (2.11 for example).
// ref: http://stackoverflow.com/questions/30342273/spark-submit-fails-with-java-lang-nosuchmethoderror-scala-predef-conformsls
//


// Get a summary of the dataset as a saddle/scala-datatable/Framian frame,
// so that dealing with this small amount of information (in comparison with the dataset) is much quicker.
//val pandas_df_training_describe = training_df.describe()


// -------------------------------------------------------------------------------------------------------
// TO-DO
// -------------------------------------------------------------------------------------------------------

// Find out which object has a std() method

//println(training_df["var15"]) //Column -> stddev ??
//println(training_df.var15) // Column ?? What is the equivalent of these two expressions in Scala?
//println(training_df.select("var15")) // DataFrame

// -------------------------------------------------------------------------------------------------------
// TO-DO
// -------------------------------------------------------------------------------------------------------

// Remove constant features

// Get std of all variables (row 2, starting from column 1) as a dict.
//  training_stds = pandas_df_training_describe.iloc[2,1:].apply(float).to_dict()

// get those columns whose std == 0.0
//remove = [col for col, value in training_stds.items() if value == 0.0]

//for col in remove:
//  training_df = training_df.drop(col)
//test_df = test_df.drop(col)


