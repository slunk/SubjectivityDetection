package edu.umd.cs.clip;

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.WeightLearningApplication
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator;
import edu.umd.cs.psl.database.DatabaseQuery;
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.resultui.UIFullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.groovy.syntax.FormulaContainer
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.model.formula.Formula;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.parameters.Weight
import com.google.common.collect.Iterables;

/*
 * Initializes DataStore, ConfigBundle, and PSLModel
 */
Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle("hier-framing")
cb.setProperty("rdbmsdatastore.usestringids", true)

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = cb.getString("dbpath", defaultPath + File.separator + "hier-framing")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)

/* Model parameters */
int minUsesOfWord = cb.getInt('minusesofword', -1);
boolean sq = cb.getBoolean('squared', true);
double initialWordWeight = cb.getDouble('initialwordweight', -1.0);
double initialSentInfluenceWeight = cb.getDouble('initialdocbiasweight', -1.0);
double initialSubsequentWeight = cb.getDouble('initialsubsequentweight', -1.0);
double initialPriorWeight = cb.getDouble('initialpriorweight', -1.0);
double variance = cb.getDouble('weightvariance', 0.0);

/* Evaluation parameters */

int folds = cb.getInt('folds', 10);
double trainTestRatio = cb.getDouble('traintestratio', 0.5);

/* Partition numbers */
Partition backgroundPart = new Partition(cb.getInt('backgroundpartition', -1));
Partition docBiasPart = new Partition(cb.getInt('docbiaspartition', -1));


PSLModel m = new PSLModel(this, data)
/*
 * Defines Predicates
 */

/* m.add predicate: "BiasPred", types: [ArgumentType.UniqueID] */
m.add predicate: "containsWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "inDoc", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
/* m.add predicate: "negated", types: [ArgumentType.UniqueID] */
m.add predicate: "hasBias", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "subsequent", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "docHasBias", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

/*
 * Loads data
 */

dataDir = cb.getString('datadir', '')

def inserter;
/* ContainsHashTag */
/* inserter = data.getInserter(BiasPred, backgroundPart);
InserterUtils.loadDelimitedData(inserter, dataDir + "bias.txt");
System.out.println("Loaded bias") */

inserter = data.getInserter(containsWord, backgroundPart);
InserterUtils.loadDelimitedData(inserter, dataDir + "words.txt");
System.out.println("Loaded containsWord")
inserter = data.getInserter(inDoc, backgroundPart);
InserterUtils.loadDelimitedData(inserter, dataDir + "doc.txt");
System.out.println("Loaded inDoc")
/* inserter = data.getInserter(negated, backgroundPart);
InserterUtils.loadDelimitedData(inserter, dataDir + "neg.txt");
System.out.println("Loaded negated") */
inserter = data.getInserter(docHasBias, docBiasPart);
InserterUtils.loadDelimitedData(inserter, dataDir + "docbias.txt");
System.out.println("Loaded docHasBias")
inserter = data.getInserter(subsequent, docBiasPart);
InserterUtils.loadDelimitedData(inserter, dataDir + "subsequent.txt");
System.out.println("Loaded subsequent sentence")

/*
 * Define Rules
 */
/* Experiment parameters */
def numIterations = cb.getInt('numiterations', -1);

/* Collects hash tag counts */
Map<GroundTerm, Integer> wordCounts = new HashMap<GroundTerm, Integer>();
DatabaseQuery query = new DatabaseQuery((containsWord(S, W).getFormula()));
Database db = data.getDatabase(backgroundPart);
ResultList results = db.executeQuery(query);
for (int iWord = 0; iWord < results.size(); iWord++) {
	GroundTerm word = results.get(iWord, W.toAtomVariable());
	if (wordCounts.containsKey(word))
		wordCounts.put(word, wordCounts.get(word) + 1);
	else
		wordCounts.put(word, 1);
}
db.close();

/* Builds set of bias terms */
Set<GroundTerm> biases = new HashSet<GroundTerm>();

db = data.getDatabase(docBiasPart);
query = new DatabaseQuery((docHasBias(Doc, Bias)).getFormula());
results = db.executeQuery(query);
for (int iResults = 0; iResults < results.size(); iResults++) {
	biases.add(results.get(iResults, Bias.toAtomVariable()));
}

System.out.println("Num biases: " + biases.size());
db.close()

Random rand = new Random(7181988);
Map<Kernel, Double> initialWeights = new HashMap<Kernel, Double>();

/* Declares bag of words rules */
for (Map.Entry<GroundTerm, Integer> word : wordCounts) {
	if (word.getValue() >= minUsesOfWord) {
		for (GroundTerm bias : biases) {
			initWeight = Math.max((initialWordWeight + rand.nextGaussian() * variance), 0.05)
			def k = m.add rule: containsWord(S, word.getKey()) >> hasBias(S, bias), weight: initWeight, squared: false
			initialWeights.put(k, initWeight);
			//k = m.add rule: (containsWord(S, word.getKey()) & negated(S)) >> hasBias(S, bias), weight: initWeight, squared: sq
			//initialWeights.put(k, initWeight);
		}
        /*k = m.add rule: (BiasPred(B) & containsWord(S, word.getKey())) >> ~hasBias(S, B), weight: initWeight, squared: false
		initialWeights.put(k, initWeight);*/
	}
}


/* Declares sentence influence rules */

initWeight = initialSentInfluenceWeight
def k = m.add rule: (hasBias(S, B) & inDoc(S, D)) >> docHasBias(D, B), weight: initWeight, squared: sq
initialWeights.put(k, initWeight);

/* Declares sentence proximity rules */

/*
initWeight = initialSubsequentWeight
k = m.add rule: (hasBias(S1, B) & subsequent(S1, S2)) >> hasBias(S2, B), weight: initWeight, squared: sq
initialWeights.put(k, initWeight);
initWeight = initialSubsequentWeight
k = m.add rule: (hasBias(S2, B) & subsequent(S2, S1)) >> hasBias(S1, B), weight: initWeight, squared: sq
initialWeights.put(k, initWeight);
*/

/* Declares prior */

m.add rule: ~hasBias(U, C), weight : 1, squared : sq

/* Declares constraints */
m.add PredicateConstraint.PartialFunctional , on : hasBias
def functionalKernel = m.add PredicateConstraint.Functional , on : docHasBias

Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel kern0 : Iterables.filter(m.getKernels(), CompatibilityKernel.class)) {
	weights.put(kern0, kern0.getWeight());
}
	
/*
 * Prepare partitions
 */
List<Partition> trainReadPartitions = new ArrayList<Partition>();
List<Partition> testReadPartitions = new ArrayList<Partition>();
List<Partition> trainEStepPartitions = new ArrayList<Partition>();
List<Partition> testEStepPartitions = new ArrayList<Partition>();
List<Partition> trainMStepPartitions = new ArrayList<Partition>();
List<Partition> testMStepPartitions = new ArrayList<Partition>();
List<Partition> trainLabelPartitions = new ArrayList<Partition>();
List<Partition> testLabelPartitions = new ArrayList<Partition>();

def keys = new HashSet<Variable>()
ArrayList<Set<Integer>> trainingSeedKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> testingSeedKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> trainingKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> testingKeys = new ArrayList<Set<Integer>>()

List<DiscretePredictionStatistics> experimentResults = new ArrayList<DiscretePredictionStatistics>();

for (int i = 0; i < folds ; i++) {
	trainReadPartitions.add(i, new Partition(i + 2));
	testReadPartitions.add(i, new Partition(i + folds + 2));
	
	trainEStepPartitions.add(i, new Partition(i + 4 * folds + 2));
	testEStepPartitions.add(i, new Partition(i + 5 * folds + 2));
	
	trainMStepPartitions.add(i, new Partition(i + 6 * folds + 2));
	testMStepPartitions.add(i, new Partition(i + 7 * folds + 2));
	
	trainLabelPartitions.add(i, new Partition(i + 2 * folds + 2));
	testLabelPartitions.add(i, new Partition(i + 3 * folds + 2));
	
	/* Generate random split */
	System.out.println("Splitting");
	List<Set<GroundTerm>> split = getRandomSplit(data, trainTestRatio, backgroundPart, docBiasPart
		, trainReadPartitions.get(i), testReadPartitions.get(i), trainLabelPartitions.get(i)
		, testLabelPartitions.get(i));
	
	Variable document = new Variable("document");
	Variable sentence = new Variable("sentence");
	Variable bias = new Variable("bias");
	FormulaContainer indoc = inDoc(sentence, document);
	FormulaContainer dochasbias = docHasBias(document, bias);
	Set<GroundTerm> sents = getGroundTerms(data, indoc, sentence, trainReadPartitions.get(i));
	Set<GroundTerm> docs = getGroundTerms(data, indoc, document, trainReadPartitions.get(i));
	biases = getGroundTerms(data, dochasbias, bias, trainLabelPartitions.get(i));
	
	Map<Variable, Set<GroundTerm>> submap = new HashMap<Variable, Set<GroundTerm>>();
	submap.put(new Variable("D"), docs);
	submap.put(new Variable("B"), biases);
	submap.put(new Variable("S"), sents);
	
	Database trainEStepDB = data.getDatabase(trainEStepPartitions.get(i));
	DatabasePopulator populator = new DatabasePopulator(trainEStepDB);
	populator.populate(hasBias(S, B).getFormula(), submap);
	trainEStepDB.close();
	System.out.println("Populated E step cluster atoms");
	
	Database trainMStepDB = data.getDatabase(trainMStepPartitions.get(i))
	populator = new DatabasePopulator(trainMStepDB);
	populator.populate(docHasBias(D, B).getFormula(), submap);
	populator.populate(hasBias(S, B).getFormula(), submap);
	trainMStepDB.close();
	System.out.println("Populated M step cluster atoms");
	
	/*toClose = [containsWord, inDoc, docHasBias] as Set;
	Database trainEStepDB = data.getDatabase(trainEStepPartitions.get(fold))
	Database testEStepDB = data.getDatabase(testEStepPartitions.get(fold))
	
	toClose = [containsWord, inDoc] as Set;
	Database trainMStepDB = data.getDatabase(trainMStepPartitions.get(fold))
	Database testMStepDB = data.getDatabase(testMStepPartitions.get(fold))*/
	eStepToClose = [containsWord, inDoc, docHasBias, subsequent] as Set
	mStepToClose =  [containsWord, inDoc, subsequent] as Set
    
	for (CompatibilityKernel kern : Iterables.filter(m.getKernels(), CompatibilityKernel.class)) {
		kern.setWeight(weights.get(kern));
	}
	
	//m.removeKernel(partialFunctionalKernel)
	
	for (int iter = 0; iter < numIterations; iter++) {
		
		db = data.getDatabase(trainEStepPartitions.get(i), eStepToClose
			, trainReadPartitions.get(i), trainLabelPartitions.get(i));
		MPEInference mpe = new MPEInference(m, db, cb);
		FullInferenceResult res = mpe.mpeInference();
		UIFullInferenceResult test = new UIFullInferenceResult(db, res);
		test.printAtoms(hasBias);
		mpe.close();
		db.close();
		
		/* M step: optimizes parameters */
		rvDB = data.getDatabase(trainMStepPartitions.get(i), mStepToClose, trainReadPartitions.get(i));
		obsvDB = data.getDatabase(trainEStepPartitions.get(i), [hasBias, docHasBias] as Set, trainLabelPartitions.get(i));
		WeightLearningApplication wl = new MaxLikelihoodMPE(m, rvDB, obsvDB, cb);
	//	WeightLearningApplication wl = new MaxPseudoLikelihood(m, rvDB, obsvDB, cb);
		wl.learn();
		wl.close();
		rvDB.close();
		obsvDB.close();
	}
    //partialFunctionalKernel = m.add PredicateConstraint.Functional , on : docHasBias
	
	sents = getGroundTerms(data, indoc, sentence, testReadPartitions.get(i));
	docs = getGroundTerms(data, indoc, document, testReadPartitions.get(i));
	
	submap = new HashMap<Variable, Set<GroundTerm>>();
	submap.put(new Variable("D"), docs);
	submap.put(new Variable("B"), biases);
	submap.put(new Variable("S"), sents);
	
	Database testMStepDB = data.getDatabase(testMStepPartitions.get(i))
	populator = new DatabasePopulator(testMStepDB);
	populator.populate(docHasBias(D, B).getFormula(), submap);
	populator.populate(hasBias(S, B).getFormula(), submap);
	testMStepDB.close();
	
	db = data.getDatabase(testMStepPartitions.get(i), mStepToClose
		, testReadPartitions.get(i));
	MPEInference mpe = new MPEInference(m, db, cb);
	FullInferenceResult res = mpe.mpeInference();
	UIFullInferenceResult test = new UIFullInferenceResult(db, res);
	test.printAtoms(hasBias);
	test.printAtoms(docHasBias);
	System.out.println("Result of inference on test set")
	mpe.close();
	
	
	def comparator = new DiscretePredictionComparator(db)
	groundTruthDB = data.getDatabase(testLabelPartitions.get(i), [docHasBias] as Set)
	comparator.setBaseline(groundTruthDB)
	comparator.setResultFilter(new MaxValueFilter(docHasBias, 1))
	comparator.setThreshold(Double.MIN_VALUE)
	
	int totalTestExamples = docs.size() * biases.size();
	System.out.println("totalTestExamples " + totalTestExamples)
	DiscretePredictionStatistics stats = comparator.compare(docHasBias, totalTestExamples)
	System.out.println("F1 score " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))

	experimentResults.add(stats);

	//DataOutputter.outputClassificationPredictions("output/" + dataSet + "/" + config.getString("name", "") + fold + ".csv", testDB, HasCat, ",")
	groundTruthDB.close();
	db.close();
}

for (DiscretePredictionStatistics stats : experimentResults) {
	System.out.println("F1 score " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
}

public Set<GroundTerm> getGroundTerms(DataStore data, FormulaContainer fml, Variable var, Partition part) {
	Set<GroundTerm> ret = new HashSet<GroundTerm>();
	DatabaseQuery query = new DatabaseQuery(fml.getFormula());
	Database db = data.getDatabase(part);
	ResultList results = db.executeQuery(query);
	for (int i = 0; i < results.size(); i++) {
		ret.add(results.get(i, var));
	}
	db.close();
	return ret;
}

public List<Set<GroundTerm>> getRandomSplit(DataStore data, double trainTestRatio, Partition backgroundPart, Partition docBiasPart
	, Partition trainPart, Partition testPart, Partition trainLabelPart, Partition testLabelPart) {
	long seed = System.nanoTime();
	List<Set<GroundTerm>> ret = new ArrayList<Set<GroundTerm>>();
	ret.add(new HashSet<GroundTerm>());
	ret.add(new HashSet<GroundTerm>());
	Set<GroundTerm> docKeys = new HashSet<GroundTerm>();
	
	Database db = data.getDatabase(backgroundPart, docBiasPart);
	DatabaseQuery query = new DatabaseQuery((docHasBias(Doc, Bias)).getFormula());
	//ResultList results = db.executeQuery(query);
	for (GroundAtom a : Queries.getAllAtoms(db, docHasBias)) {
		docKeys.add(a.getArguments()[0]);
		//System.out.println(a.getArguments()[0]);
	}
	db.close();
	
	List<GroundTerm> docKeyList = new ArrayList<GroundTerm>();
	docKeyList.addAll(docKeys);
	Collections.shuffle(docKeyList, new Random(seed))
	Set<GroundTerm> trainDocKeys = new HashSet<GroundTerm>();
	Set<GroundTerm> testDocKeys = new HashSet<GroundTerm>();
	
	for (int i = 0; i < docKeyList.size(); i++) {
		if (i <= trainTestRatio * docKeyList.size()) {
			trainDocKeys.add(docKeyList.get(i));
		} else {
			testDocKeys.add(docKeyList.get(i));
		}
	}
	
	Set<GroundTerm> trainSentKeys = new HashSet<GroundTerm>();
	Set<GroundTerm> testSentKeys = new HashSet<GroundTerm>();
	
	def trainInserter = data.getInserter(inDoc, trainPart);
	def testInserter = data.getInserter(inDoc, testPart);
	db = data.getDatabase(backgroundPart, docBiasPart);
	for (GroundAtom a : Queries.getAllAtoms(db, inDoc)) {
		if (trainDocKeys.contains(a.getArguments()[1])) {
			trainSentKeys.add(a.getArguments()[0]);
			trainInserter.insertValue(a.getValue(), a.getArguments());
		} else {
			testSentKeys.add(a.getArguments()[0]);
			testInserter.insertValue(a.getValue(), a.getArguments());
		}
	}
	
	trainInserter = data.getInserter(containsWord, trainPart);
	testInserter = data.getInserter(containsWord, testPart);
	for (GroundAtom a : Queries.getAllAtoms(db, containsWord)) {
		if (trainSentKeys.contains(a.getArguments()[0])) {
			trainInserter.insertValue(a.getValue(), a.getArguments());
		} else {
			testInserter.insertValue(a.getValue(), a.getArguments());
		}
	}

	trainInserter = data.getInserter(negated, trainPart);
	testInserter = data.getInserter(negated, testPart);
	for (GroundAtom a : Queries.getAllAtoms(db, negated)) {
		if (trainSentKeys.contains(a.getArguments()[0])) {
			trainInserter.insertValue(a.getValue(), a.getArguments());
		} else {
			testInserter.insertValue(a.getValue(), a.getArguments());
		}
	}
	
	trainInserter = data.getInserter(subsequent, trainPart);
	testInserter = data.getInserter(subsequent, testPart);
	for (GroundAtom a : Queries.getAllAtoms(db, subsequent)) {
		if (trainSentKeys.contains(a.getArguments()[0])) {
			trainInserter.insertValue(a.getValue(), a.getArguments());
		} else {
			testInserter.insertValue(a.getValue(), a.getArguments());
		}
	}
	db.close()
	
	db = data.getDatabase(backgroundPart, docBiasPart);
	trainInserter = data.getInserter(docHasBias, trainLabelPart);
	testInserter = data.getInserter(docHasBias, testLabelPart);
	for (GroundAtom a : Queries.getAllAtoms(db, docHasBias)) {
		if (trainDocKeys.contains(a.getArguments()[0])) {
			trainInserter.insertValue(a.getValue(), a.getArguments());
		} else {
			testInserter.insertValue(a.getValue(), a.getArguments());
		}
	}
	db.close();
	
	System.out.println(trainDocKeys.size());
	System.out.println(testDocKeys.size());
	System.out.println(trainSentKeys.size());
	System.out.println(testSentKeys.size());
	return ret;
}
