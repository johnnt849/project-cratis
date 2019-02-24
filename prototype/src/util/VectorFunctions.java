package util;

import java.lang.Math;
import java.util.ArrayList;

import util.DimensionMismatchException;

public class VectorFunctions {
	// ====== VECTOR OPERATIONS ======
	public static ArrayList<Double> multiplyByScalar(double scalar, ArrayList<Double> vector) {
		ArrayList<Double> result = new ArrayList<Double>(vector.size());
		for (double d: vector) result.add(scalar * d);

		return result;
	}

	public static ArrayList<Double> tanhVectorActivation(ArrayList<Double> vector) {
		ArrayList<Double> result = new ArrayList<Double>(vector.size());
		for (double d: vector) result.add(Math.tanh(d));

		return result;
	}

	public static ArrayList<Double> elementwiseMultVectors(ArrayList<Double> vec1, ArrayList<Double> vec2) {
		if (vec1.size() != vec2.size()) throw new DimensionMismatchException("Vectors are not the same size");

		ArrayList<Double> result = new ArrayList<Double>(vec1.size());
		for (int i = 0; i < vec1.size(); i++) result.add(vec1.get(i) * vec2.get(i));

		return result;
	}

	public static double multiplyAndSumVectors(ArrayList<Double> vec1, ArrayList<Double> vec2) {
		if (vec1.size() != vec2.size()) throw new DimensionMismatchException("Vectors are not the same size");

		double result = 0;
		for (int i = 0; i < vec1.size(); i++) result += vec1.get(i) * vec2.get(i);

		return result;
	}

	public static ArrayList<Double> sumVectors(ArrayList<Double> vec1, ArrayList<Double> vec2) {
		if (vec1.size() != vec2.size()) throw new DimensionMismatchException("Vectors are not the same size");
		ArrayList<Double> result = new ArrayList<Double>(vec1.size());
		for (int i = 0; i < vec1.size(); i++) {
			result.add(vec1.get(i) + vec2.get(i));
		}

		return result;
	}

	public static double sum(ArrayList<Double> vector) {
		double result = 0;
		for (double d: vector) result += d;

		return result;
	}

	// ====== NEURAL NETWORK FUNCTIONS ======
	// Output Evaluation
	/**
	* Converts an array of output activations to precidctions of which
	* class the Vertex is in (all rows sum to 1)
	* @param ArrayList<Double> vector		Activation vector corresponding to vertex
	* @return ArrayList<Double> 
	*/
	public static ArrayList<Double> softmax(ArrayList<Double> vector) {
		ArrayList<Double> result = new ArrayList<Double>(vector.size());

		for (double d: vector) result.add(Math.exp(d));
		double total = sum(result);

		result.replaceAll(d -> d / total);
		return result;
	}

	/**
	* Predicts the output loss for one Vertex given its softmax predictions and 
	* its classification
	* @param ArrayList<Double>		Predictions produced by softmax
	* @return double				Loss for specific Vertex
	*/
	public static double cross_entropy(ArrayList<Double> preds, int classification) {
		// This will only work for BINARY classification
		return (-Math.log(preds.get(classification)));
	}

	// Backpropagation
	/**
	* Derivative of the activation function
	*/
	public static double derivTanh(double d) {
		return (1 - Math.pow(Math.tanh(d), 2));
	}

	/**
	* Apply the derivative of the activation to all elements in a Vertex activation
	*/
	public static ArrayList<Double> activationPrime(ArrayList<Double> activations) {
		ArrayList<Double> result = new ArrayList<Double>(activations.size());
		for (double d: activations) result.add(derivTanh(d));

		return result;
	}

	/**
	* Apply the derivate of the cross entropy with softmax function to the output and divide
	* by batch size
	* Deriv of cross ent w/ softmax: p_i - y_i 
	* @param trainSize		The number of training examples
	*/
	public static ArrayList<Double> deltaCrossEntropy(ArrayList<Double> activations, int classification, int trainSize) {
		ArrayList<Double> result = softmax(activations);
		result.set(classification, result.get(classification) - 1);

		result.replaceAll(d -> (d / trainSize));
		return result;
	}


	// ====== UTILITY FUNCTION ======
	public static void printVector(ArrayList<Double> vector) {
		for (double d: vector) {
			System.out.print(Double.toString(d) + " ");
		}
		System.out.println();
	}
}

