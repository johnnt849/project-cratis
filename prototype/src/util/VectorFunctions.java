package util;

import java.lang.Math;
import java.util.ArrayList;

import util.DimensionMismatchException;

public class VectorFunctions {
	// VECTOR OPERATIONS
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

	// NEURAL NETWORK FUNCTIONS
	public static ArrayList<Double> softmax(ArrayList<Double> vector) {
		ArrayList<Double> result = new ArrayList<Double>(vector.size());

		for (double d: vector) result.add(Math.exp(d));
		double total = sum(result);

		result.replaceAll(d -> d / total);
		return result;
	}


	// UTILITY FUNCTION
	public static void printVector(ArrayList<Double> vector) {
		for (double d: vector) {
			System.out.print(Double.toString(d) + " ");
		}
	}
}

