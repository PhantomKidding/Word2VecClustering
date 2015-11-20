

package myspark

import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors


class Word2Vec extends org.apache.spark.mllib.feature.Word2Vec {

	var WORD2VEC_SEQUENCE_FILE_NAME = "sequence.file.0";
	var WORD2VEC_MATRIX_FILE = "word2vecMatrix";

	private var _SPARK_CONTEXT: SparkContext = null;
	private var _INPUT_DIR = "input/";
	private var _OUTPUT_DIR = "output/word2vec/";
	private var _NUM_SYNONYMS = 50;
	private var _MIN_SIMILARITY = 20;
	private var _IF_CREATE_SEQUENCE_FILE = true;
	private var _WORD_SEQUENCE: Seq[String] = null;

	def train() {
	  if(!new File(_OUTPUT_DIR).exists())
	    new File(_OUTPUT_DIR).mkdirs();
	  
		val input = _SPARK_CONTEXT.textFile(_INPUT_DIR)
				.map(x ⇒ x.split(" ").toSeq);

		val word2vecObject = new Word2Vec();
		val model = word2vecObject.fit(input);


		val totalWords = model.getVectors.toSeq.sortWith(_._1 < _._1);
		val totalWordsSize = totalWords.size;
		val totalWordsSeq = totalWords.map(x ⇒ x._1);
		_WORD_SEQUENCE = totalWordsSeq;
		
		if(_IF_CREATE_SEQUENCE_FILE)
			createSequence(totalWordsSeq);

		
		val simMatrix = totalWordsSeq.map(word ⇒
				Vectors.sparse(totalWordsSize,
						model.findSynonyms(word, totalWordsSize)
//						.filter(x ⇒ x._2 > _MIN_SIMILARITY)
						.map(x ⇒ (totalWordsSeq.indexOf(x._1), x._2))
						.toSeq));
//		val simMatrix = totalWordsSeq.map(word ⇒
//				Vectors.sparse(totalWordsSize,
//						model.findSynonyms(word, _NUM_SYNONYMS)
//						.map(x ⇒ (totalWordsSeq.indexOf(x._1), x._2))
//						.toSeq));
		val writer = new BufferedWriter(new FileWriter(new File(_OUTPUT_DIR + WORD2VEC_MATRIX_FILE)));
		simMatrix.foreach(x ⇒ writer.write(x.toDense.toString().replaceAll("\\[|\\]", "").replaceAll(",", " ") + "\n"));
		writer.close();
	}
	
	def createSequence(input: Seq[String]) {
		val writer = new BufferedWriter(new FileWriter(new File(_OUTPUT_DIR + WORD2VEC_SEQUENCE_FILE_NAME)));
		input.foreach(x ⇒ writer.write(x + "\n"));
		writer.close()
	}
	
	def setSparkContext(sc: SparkContext): this.type = {this._SPARK_CONTEXT = sc; this }
	def setInput(input: String): this.type = { this._INPUT_DIR = input; this }
	def setOutput(output: String): this.type = { this._OUTPUT_DIR = output; this }
	def setNumSynonyms(k: Int): this.type = { this._NUM_SYNONYMS = k; this }
	def createSeqFile(ts: Boolean): this.type = { this._IF_CREATE_SEQUENCE_FILE = ts; this }
	
	def getOutput(): String = _OUTPUT_DIR;
	def getWordSeq(): Seq[String] = _WORD_SEQUENCE;
}