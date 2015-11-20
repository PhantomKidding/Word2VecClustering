package myspark

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.rdd.RDD
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

class Kmeans {

  var KMEANS_TOPIC_FILE = "topics";
  
  private var _SPARK_CONTEXT: SparkContext = null;
  private var _INPUT = "input/";
  private var _OUTPUT = "output/";
  private var _K = 2;
  private var _EPOCH = 20;
  private var _NUM_NEAREST_NODES = 1;
  private var _WORD_SEQUENCE: Seq[String] = null;
  
  def train(): Array[Array[Long]] = {
    println(_WORD_SEQUENCE == null);
    if(!new File(_OUTPUT).exists())
	    new File(_OUTPUT).mkdirs();
    
	  val parsedData = _SPARK_CONTEXT.textFile(_INPUT)
			  .map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache();
	  
    val model = KMeans.train(parsedData, _K, _EPOCH);
    val centroids = model.clusterCenters;
    val topics = centroids.map(x ⇒ closestNNodes(x, parsedData, _NUM_NEAREST_NODES));
    saveTopics(topics);
    centroids.foreach { x => println(x.toDense) };
    topics;
  }
  
  def saveTopics(input: Array[Array[Long]]) {
	  val writer = new BufferedWriter(new FileWriter(new File(_OUTPUT + KMEANS_TOPIC_FILE)));
	  if(_WORD_SEQUENCE == null) {
		  input.foreach(x ⇒ 
				  for(i ← 0 until x.length) { 
					  writer.write(x.apply(i) + (if(i == (x.length - 1)) "\n" else ",")) 
				  });
	  } else {
		  input.foreach(x ⇒ 
				  for(i ← 0 until x.length) { 
					  writer.write(_WORD_SEQUENCE.apply(x.apply(i).toInt) + (if(i == (x.length - 1)) "\n" else ",")) 
				  })
	  }
	  writer.close();
  }
  
  def closestNNodes(centroid: Vector, data: RDD[Vector], n: Int): Array[Long] = {
		  val orderedData = data.map(x ⇒ EuclideanDistance(centroid, x)).zipWithIndex()
				  .takeOrdered(n)(Ordering.by(-1 * _._1));
		  orderedData.map(x ⇒ x._2);
  }
  
  def EuclideanDistance(a: Vector, b: Vector): Double = {
    var sum: Double = 0.0;
    var i: Int = 0;
    for(i ← 0 until a.size)
    	sum += Math.pow((a.apply(i) - b.apply(i)), 2);
    Math.sqrt(sum);
    return(Math.sqrt(sum));
  }
   
  def setSparkContext(sc: SparkContext): this.type = { this._SPARK_CONTEXT = sc; this }
  def setInput(input: String): this.type = { this._INPUT = input; this }
  def setOutput(output: String): this.type = { this._OUTPUT = output; this }
  def setNumCluster(k: Int): this.type = { this._K = k; this }
  def setNumIteration(n: Int): this.type = { this._EPOCH = n; this }
  def setNumNearestNodes(n: Int): this.type = { this._NUM_NEAREST_NODES = n; this }
  def setWordSequence(seq: Seq[String]): this.type = { this._WORD_SEQUENCE = seq; this }
  def setWordSequence(seq: String): this.type = { 
    this._WORD_SEQUENCE = _SPARK_CONTEXT.textFile(seq).collect().toSeq; this }
  
  def getNumCluster(): Int = this._K;
}