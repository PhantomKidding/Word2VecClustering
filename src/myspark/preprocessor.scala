
package myspark

import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapred.lib.MultipleTextOutputFormat
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions



object preprocessor {

	val RAW_INPUT_DIR = "raw/";
	val PREPROCESSOR_OUTPUT_DIR = "nips/";

	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("nips").setMaster("local");
		val sc = new SparkContext(conf);

		val seq = sc.textFile(RAW_INPUT_DIR + "vocab.nips.txt").collect();
		val raw = sc.textFile(RAW_INPUT_DIR + "docword.nips.txt")
				.map(x => x.split(" "))
				.filter(x => x.size == 3)
				.map(x ⇒ (x.apply(0), join(seq.apply(x.apply(1).toInt - 1), x.apply(2))))
				.reduceByKey(_ + " " + _)
				//				.map(x ⇒ (null, x._2));
				//		raw.saveAsNewAPIHadoopFile(
				//				PREPROCESSOR_OUTPUT_DIR, 
				//				classOf[NullWritable], 
				//				classOf[Text], 
				//				classOf[TextOutputFormat[NullWritable, Text]],
				//				sc.hadoopConfiguration);
				raw.saveAsHadoopFile(PREPROCESSOR_OUTPUT_DIR, 
						classOf[String], 
						classOf[String], 
						classOf[RDDMultipleTextOutputFormat]);
	}

	private def join(input: (String, String)): String = {
		val w = input._1;
		val n = input._2.toInt;
		var output: String = w.toString();
		var i = 0;
		for(i ← 2 to n) {
			output += " " + w.toString()
		}
		return(output)
	}
}

class RDDMultipleTextOutputFormat extends MultipleTextOutputFormat[Any, Any] {
	override def generateActualKey(key: Any, value: Any): Any = {
			NullWritable.get() }

	override def generateFileNameForKeyValue(key: Any, value: Any, name: String): String = {
			"part-" + key.asInstanceOf[String] }
}
