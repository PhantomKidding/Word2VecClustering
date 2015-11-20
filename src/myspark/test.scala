
package myspark

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

object test {
	def main(args: Array[String]) {

		val conf = new SparkConf().setAppName("test").setMaster("local");
		val sc = new SparkContext(conf);

		val word2vec = new Word2Vec().setSparkContext(sc)
//				.setInput(preprocessor.PREPROCESSOR_OUTPUT_DIR + "part-r-00000")
				.setInput("nips/part-*")
				.setOutput("output/word2vec/")
				.setNumSynonyms(1000)
//				word2vec.train();

		val kmeansInit = new Kmeans() with Serializable;
		val kmeans = kmeansInit.setSparkContext(sc)
				.setInput("output/word2vec/" + word2vec.WORD2VEC_MATRIX_FILE)
				.setOutput("output/kmeans/")
				.setNumCluster(5)
				.setNumIteration(1000)
				.setNumNearestNodes(10)
				//				.setWordSequence(word2vec.getWordSeq());
				.setWordSequence("output/word2vec/sequence.file.0");
		kmeans.train();

		
		val compareKmeansInit = new KMeans();
		val data = sc.textFile("nips/part-*");
		val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache();
		val clusters = KMeans.train(parsedData, 5, 1000);
		val centroids = clusters.clusterCenters;
		val writer = new BufferedWriter(new FileWriter(new File("output/compareK")))
		centroids.foreach(x ⇒ for(i ← 0 until x.toArray.length) { 
		  writer.write(x.toArray.apply(i) + (if(i == x.toArray.length) "\n" else ","))
		});
//		val compareKmeansInit = new Kmeans() with Serializable;
//		val compareKmeans = kmeansInit.setSparkContext(sc)
//				.setInput("nips/part-r-00000")
//				.setOutput("output/compareK")
//				.setNumCluster(25)
//				.setNumIteration(1000)
//				.setNumNearestNodes(20)
				//	  val input = sc.textFile("output/word2vec/sequence.file.0").zipWithIndex();
				//	  val a = Seq(input.map(x ⇒ x._2).collect());
				//	  a.foreach(x ⇒ for(i ← 0 until x.length) { 
				//    		print(x.apply(i) + (if(i == (x.length - 1)) "\n" else "\n")) 
				//    	})
	}
}