
NOTE: Scala 2.10.5 version is used, so that it is compatible with Spark.

Now I use IntelliJ IDEA, and the Jupyter Notebook works. I just has to go 
to File>Project Structure>Project SDK and choose the /anaconda/Python one that is installed in your system.
Now I have to install the IScala plugin for jupyter, 
so that I can run Scala scripts with IntelliJ IDEA. -> I have tried this, but the only versions available 
are opensource and I am having problems when trying to install them. So I will just use plain Scala scripts 
with IntelliJ IDEA.

The key to use several languages in the same IntelliJ IDEA projects is the concept of module.
A module is a folder containing an .iml file which defines the compile/interpreter tool to be used within that module.
I didn't need to define the previous folder (./4.Santander_Customer_satisfaction) as a module, as the main SDK is 
preset to python in the Project Structure, so this setting is ok for both plain python and pyspark, which is 
just a python library. However, to run Scala-Spark scripts, we need to define the SDK to be used for this script, 
so a new folder containing an .iml file (a.k.a. module) must be defined: "5.Scala_Customer_Satisfaction". 
Now this module contains the script ScalaSpark.scala, which can be quicky run.

Anyways, next time a new project is created, define a module for each language used.


When trying to run spark on windows, I get this error: 

http://stackoverflow.com/questions/32721647/why-does-starting-spark-shell-fail-with-nullpointerexception-on-windows

The solution is the most voted answer, in case you also have this error.


Now, to include Spark (and in general any .jar) do the following steps:

Click File -> Project Structure. Under Libraries add 
{SPARK_HOME}\lib\spark-assembly-1.6.1-hadoop2.6.0.jar

(more info at http://hackers-duffers.logdown.com/posts/245018-configuring-spark-with-intellij)

Another alternative is to use maven to add libraries (Project Structure > Libraries / Global Libraries > add from Maven) 
When doing this, note some libraries may clash, e.g., if Scala SDK uses its 2.10.5 version and spark-core/spark-csv/framian 
are added to the libraries from maven, they may contain a different version of Scala, which might class with the first one. 
The only solution I found is to manually remove the scala jars contained into this packages by doing: 
Project Structure > Libraries/Global libraries (depending on where they were added > *remove symbol* on scala-compiler/library/assembly, etc..
But it still has errors. 

So the approach, at least for Spark, is to add spark-assembly, not spark-core, and then adding more libraries. In summary:

Add spark-assembly-1.6.1-hadoop2.6.0 - from here:
· databricks-csv 2.11:1.4.0 - compatible
· net.tixxit:framian_2.11:0.5.0 - compatible

IMPORTANT! if you first import databricks-csv and framian, and then spark-assembly, there will be clashing errors. Why?
The thing is, these dependencies are managed by intelliJ IDEA, and they are loaded in a preset order. The project needs to 
load spark-assembly first, then databricks and framian. To change this order, go to Project Structure > Modules and 
move spark-assembly up. Now run your script. It works!
