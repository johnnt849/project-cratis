package engine;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

import data.Vertex;
import data.Weights;
import engine.Mapper;
import engine.Preprocess;
import engine.Reducer;
import util.Matrix;
import util.MatrixImpl;
import util.Vector;
import util.VectorFunctions;
import util.VectorImpl;



public class GCN {
	ArrayList<Vertex> graph;
	ArrayList<Weights> weights;

	int nMappers;
	int chunkSize;

	double eta;

	public GCN(ArrayList<Vertex> g, int nMaps, int chunks) {
		graph = g;
		weights = new ArrayList<Weights>(3);
		weights.add(new Weights("../res/weights0"));
		weights.add(new Weights("../res/weights1"));
		weights.add(new Weights("../res/weights2"));

		nMappers = nMaps;
		chunkSize = chunks;

		eta = .001
	}

	/**
	* Given the processed graph, find a train mask,
	* run forward propagation and calculate error,
	* then run back propagation
	*/
	public void run(ArrayList<Vertex> graph, int numIters) {
		Integer[] trainMask = getTrainMask(graph);
		Integer[] valMask = getValMask(graph.size(), trainMask);

		// Forward pass
		for (int i = 0; i < numIters; i++) {
			forwardProp(graph);
			
			if (i % 50 == 0) {
				double error = calcTotalError(graph, valMask);
				System.out.println("Total error for iteration " + Integer.toString(i) + ": " + Double.toString(error));
			}

			List<Vertex> trainPoints = new ArrayList<Vertex>(trainMask.length);
			for (int ind: trainMask) trainPoints.add(graph.get(ind));
			List<Matrix> updates = backProp(trainPoints);

			for (Matrix m: updates) {
				for (int i = 0; i < m.get(i); i++) {
					
				}
			}
		}
	}

	/**
	* Find one random Vertex per class to use for training
	*/
	private Integer[] getTrainMask(ArrayList<Vertex> graph) {
		ArrayList<Integer> class0Indices = new ArrayList<Integer>();
		ArrayList<Integer> class1Indices = new ArrayList<Integer>();
		for (int ind = 0; ind < graph.size(); ind++) {
			if (graph.get(ind).getClassification() == 0) class0Indices.add(ind);
			else class1Indices.add(ind);
		}

		int class0Ind = class0Indices.get(((int)(Math.random() * (class0Indices.size() * 2))) % class0Indices.size());
		int class1Ind = class1Indices.get(((int)(Math.random() * (class1Indices.size() * 2))) % class1Indices.size());

		Integer[] indices = new Integer[]{class0Ind, class1Ind};
		for (int ind: indices) {
			System.out.print(Integer.toString(ind) + " ");
			System.out.println(graph.get(ind).getClassification());
		}
	
		return indices;
	}

	/**
	* Use the training mask to get the validation mask (training maks's complement)
	* @param Integer[] trainMask		The list of indices of the training points
	* @param int dataSize				The total number of vertices
	* @return Integer[]					The list of indices of validation points
	*/
	public Integer[] getValMask(int dataSize, Integer[] trainMask) {
		ArrayList<Integer> valMask = new ArrayList<Integer>(dataSize-2);
		for (int ind = 0; ind < dataSize; ind++) {
			if (ind != trainMask[0] && ind != trainMask[1]) 
				valMask.add(ind);
		}

		return valMask.toArray(new Integer[]{});
	}

	/**
	* Aggregates the weighted features of each vertices neighbors
	*
	* @param ArrayList<Vertex> g
	* @return ArrayList<Vertex>
	*/
	private Matrix aggregateFeatures(ArrayList<Vertex> graph) {
		Matrix aggFeats = new MatrixImpl(graph.size());
		for (Vertex v: graph) {
			aggFeats.add(v.getNormalizedNeighborFeatures());
		}

		return aggFeats;
	}

	/**
	* Run the forward propagation for one layer of the neural network
	*
	* @param ArrayList<Vertex> graph
	* @param int layer
	*/
	private void propagateForwardOneLayer(ArrayList<Vertex> graph, int layer) {
		Matrix aggFeats = aggregateFeatures(graph);
		Weights w = weights.get(layer);

		// Z = H * W
		for (int i = 0; i < graph.size(); i++) {
			Vector z = new VectorImpl(w.getNumRows());
			for (int j = 0; j < w.getNumRows(); j++) {
				z.add(VectorFunctions.multiplyAndSumVectors(aggFeats.get(i), w.get(j)));
			}
			Vertex v = graph.get(i);
			v.addZ(z);
			
			// tanh(Z)
			v.addActivation(VectorFunctions.tanhVectorActivation(z));
		}
	}
	
	/**
	* Run all iterations of forward propagation to the final output
	*
	* @param ArrayList<Vertex> graph		List of all vertices
	*/
	private void forwardProp(ArrayList<Vertex> graph) {
		for (int i = 0; i < weights.size(); i++) {
			propagateForwardOneLayer(graph, i);
			printActivations(graph, i+1);
		}

//		System.out.println("\nSOFTMAX PREDICTIONS");
//		for (Vertex v: graph) {
//			VectorFunctions.printVector(VectorFunctions.softmax(v.getCurrentActivations()));
//			System.out.println();
//		}
	}

	/**
	* Use the MapReduce model to compute the final gradient changes for a given layer
	* dW_l = activations_l-1^T * delta_l
	* 
	* @param ArrayList<Vertex> graph
	*/
	private Matrix getGradients(List<Vertex> trainPoints, int layer) {
		// This code is messy. Needs cleanup
		int nReducers = weights.get(layer).getNumRows();	// One reducer per row in Weight matrix

		// Initialize shared memory: for MapReduce communication
		ArrayList< ArrayList<Vector> > buckets = new ArrayList< ArrayList<Vector> >(nReducers);
		Matrix gradients = new MatrixImpl(nReducers);
		for (int i = 0; i < nReducers; i++) {
			buckets.add(new ArrayList<Vector>());
			gradients.add(new VectorImpl());
		}	// end Iniitialize shared memory

		// Initialize Mappers: allocate partitions and get vectors
		List<Mapper> mappers = new ArrayList<Mapper>(nMappers);
		for (int i = 0; i < nMappers; i++) {
			List<Vector> activationChunks = new ArrayList<Vector>(chunkSize);
			List<Vector> deltaChunks = new ArrayList<Vector>(chunkSize);

			int end = i*chunkSize+chunkSize;
			end = end < trainPoints.size() ? end : trainPoints.size();

			for (int ind = i*chunkSize; ind < end; ind++) {
				Vertex v = trainPoints.get(ind);
				deltaChunks.add(v.getCurrentDelta());
				activationChunks.add(v.getActivations(layer));
			}

			Mapper map = new Mapper(buckets, deltaChunks, activationChunks);
			mappers.add(map);
			map.start();
		}	// end Initialize Mappers

		// Initialize Reducers
		List<Reducer> reducers = new ArrayList<Reducer>(nReducers);
		for (int i = 0; i < nReducers; i++) reducers.add(new Reducer(buckets, i, gradients));
		// end Initialize Reducers

		// Wait for mappers to finish work
		for (Mapper m: mappers) {
			try {
				m.join();
			} catch(Exception e) {}
		}


		for (Reducer r: reducers) {
			r.start();
		}

		for (Reducer r: reducers) {
			try {
				r.join();
			} catch(Exception e) {}
		}
		
		
		return gradients;
	}

	private List<Matrix> backProp(List<Vertex> trainPoints) {
		List<Matrix> weightGradients = new ArrayList<Matrix>(weights.size());

		//find output delta
		for (Vertex v: trainPoints) {
			Vector lossPrime = VectorFunctions.deltaCrossEntropy(v.getCurrentActivations(), v.getClassification(), trainPoints.size());
			Vector activationPrime = VectorFunctions.activationPrime(v.getCurrentZ());
			Vector outputDelta = VectorFunctions.elementwiseMultVectors(lossPrime, activationPrime);
			v.addDelta(outputDelta);
		}

		Matrix weightChangesForLayer = getGradients(trainPoints, weights.size()-1);
		weightGradients.add(weightChangesForLayer);

		// POTENTIAL for parallelization in these two sections
		// find previous deltas
		for (int layer = weights.size()-1; layer > 0; layer--) {
			Weights w = weights.get(layer);
			for (Vertex v: trainPoints) {
				Vector activationPrime = VectorFunctions.activationPrime(v.getZ(layer-1));

				// use activations to calc weight changes
				// MapReduce section
				Vector deltaDotWT = new VectorImpl(w.getNumRows());
				for (int i = 0; i < w.getNumCols(); i++) {
					deltaDotWT.add(VectorFunctions.multiplyAndSumVectors(v.getCurrentDelta(), w.getT(i)));
				}

				Vector delta = VectorFunctions.elementwiseMultVectors(deltaDotWT, activationPrime);
				v.addDelta(delta);
			}

			weightChangesForLayer = getGradients(trainPoints, layer-1);
			weightGradients.add(weightChangesForLayer);
		}

		return weightGradients;
	}

	/**
	* Calculate the cross entropy error for a given Vertex output
	* @param Vector output		The final activations for a given Vertex
	* @param int classification				The classification of the given Vertex
	*/
	private double calcError(Vector output, int classification) {
		return VectorFunctions.cross_entropy(VectorFunctions.softmax(output), classification);
	}

	/**
	* Calculate the average error for all vertices in a mask
	* @param ArrayList<Vertex> graph		The list of all Vertex
	* @param Integer[] mask					The indices of vertices to use
	*/
	private double calcTotalError(ArrayList<Vertex> graph, Integer[] mask) {
		double error = 0;
		for (Integer ind: mask) {
			Vertex v = graph.get(ind);
			error += calcError(v.getCurrentActivations(), v.getClassification());
		}

		return error / mask.length;
	}

	/**
	* Utility function to print the graph
	* @param ArrayList<Vertex> g
	*/
	public void printGraph(ArrayList<Vertex> graph) {
		String res = "";
		for (Vertex v: graph) {
			res += v.toString() + "\n";
		}
		 res += "\n";

		System.out.print(res);
	}


	/**
	* Print the activation values for a specific layer
	*/
	public void printActivations(ArrayList<Vertex> graph, int layer) {
		System.out.println("Activation for layer " + Integer.toString(layer));
		for (Vertex v: graph) {
			System.out.print(Integer.toString(v.getVertexId()) + "| ");
			v.printActivations(layer);
		}
		System.out.println();
	}
}
