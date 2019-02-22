package engine;

import java.util.Arrays;
import java.util.ArrayList;

import data.Vertex;
import data.Weights;
import engine.Preprocess;
import util.VectorFunctions;


public class GCN {
	ArrayList<Vertex> graph;

	ArrayList<Weights> weights;

	public GCN(ArrayList<Vertex> g) {
		graph = g;
		weights = new ArrayList<Weights>(3);
		weights.add(new Weights(2, 4));
		weights.add(new Weights(4, 4));
		weights.add(new Weights(4, 2));
	}

	/**
	* Given the processed graph, find a train mask,
	* run forward propagation and calculate error,
	* then run back propagation
	*/
	public void run(ArrayList<Vertex> graph) {
		Integer[] trainMask = getTrainMask(graph);
		Integer[] valMask = getValMask(graph.size(), trainMask);

		// Forward pass
		int numIters = 1;
		for (int i = 0; i < numIters; i++) {
			forwardProp(graph);
			
			if (i % 50 == 0) {
				double error = calcTotalError(graph, valMask);
				System.out.println("Total error for iteration " + Integer.toString(i) + ": " + Double.toString(error));
			}

			backProp(graph, trainMask);
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
	private ArrayList< ArrayList<Double> > aggregateFeatures(ArrayList<Vertex> graph) {
		ArrayList< ArrayList<Double> > aggFeats = new ArrayList< ArrayList<Double> >(graph.size());
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
		ArrayList< ArrayList<Double> > aggFeats = aggregateFeatures(graph);
		Weights w = weights.get(layer);

		// Z = H * W
		for (int i = 0; i < graph.size(); i++) {
			ArrayList<Double> z = new ArrayList<Double>(w.getNumRows());
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

		System.out.println("\nSOFTMAX PREDICTIONS");
		for (Vertex v: graph) {
			VectorFunctions.printVector(VectorFunctions.softmax(v.getCurrentActivations()));
			System.out.println();
		}
	}

	private void backProp(ArrayList<Vertex> graph, Integer[] trainMask) {
		//find output delta
		Delta outputDelta = new Delta(3);
		for (int ind: trainMask) {
			Vertex v = graph.get(ind);
			outputDelta.addDelta(VectorFunctions.elementwiseMultVectors(VectorFunctions.deltaCrossEntropy(v.getCurrentActivations(), v.getClassification(), trainMask.size()), VectorFunctions.activationPrime(v.getCurrentActivations())));
		}

		// POTENTIAL for parallelization in these two sections
		// find previous deltas
		// use activations to calc weight changes
	}

	/**
	* Calculate the cross entropy error for a given Vertex output
	* @param ArrayList<Double> output		The final activations for a given Vertex
	* @param int classification				The classification of the given Vertex
	*/
	private double calcError(ArrayList<Double> output, int classification) {
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
