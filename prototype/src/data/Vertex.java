package data;

import java.lang.Math;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;

import util.VectorFunctions;

public class Vertex {
	int vertexId;

	double normalization;

	int classification;
	ArrayList<Double> features;		// Initial vertex features

	ArrayList< ArrayList<Double> > zs;	// output of weights * features BEFORE activation
	ArrayList< ArrayList<Double> > activations;

	ArrayList<Integer> edgeIndices;	// the outgoing vertexId and index into the graph
	ArrayList<Double> edgeWeights;	// normalization along the edge
	ArrayList<Vertex> graph;

	public Vertex(int vId, ArrayList<Vertex> _g, int _class, ArrayList<Double> feats) {
		vertexId = vId;
		graph = _g;

		classification = _class;
		features = feats;

		zs = new ArrayList< ArrayList<Double> >();
		activations = new ArrayList< ArrayList<Double> >();
		activations.add(feats);

		edgeIndices = new ArrayList<Integer>();
		edgeWeights = new ArrayList<Double>();
	}

	/**
	* Set the normalization for this Vertex
	*/
	public void setNormalization() {
		double degree = (double)edgeIndices.size();
		normalization = Math.pow(degree, -0.5);
	}

	/**
	* Insert an edge in sorted order
	* @param int dst	The edge to be added
	*/
	public void insertEdge(int dst) {
		for (int i = 0; i < edgeIndices.size(); i++) {
			if (dst < edgeIndices.get(i)) {
				edgeIndices.add(i, dst);
				return;
			} else if (dst == edgeIndices.get(i)) {
				return;
			}
		}
		edgeIndices.add(dst);
	}

	/**
	* Add a current layer to the zs list
	*/
	public void addZ(ArrayList<Double> z) {
		zs.add(z);
	}

	/**
	* Add a new set of activations for this vertex
	*/
	public void addActivation(ArrayList<Double> act) {
		activations.add(act);
	}

	// getters
	public int getVertexId() { return vertexId; }
	public int getClassification() { return classification; }
	public ArrayList<Double> getFeatures(int layer) { return activations.get(layer); }
	public ArrayList<Double> getInputFeatures() { return features; }
	public ArrayList<Double> getCurrentFeatures() { return activations.get(activations.size()-1); }
	public ArrayList<Double> getZ(int layer) { return zs.get(layer); }
	public int getDegree() { return edgeIndices.size(); }
	public double getNormalization() { return normalization; }
	public double getEdgeWeight(int vId) { return edgeWeights.get(edgeIndices.indexOf(vId)); }


	/**
	* Add a weight for each edge based on the normalization of the current Vertex
	* and it's outgoing neighbor
	*/
	public void findEdgeNormalization() {
		for (int i = 0; i < edgeIndices.size(); i++) {
			double dstNormalizationFactor = graph.get(edgeIndices.get(i)).getNormalization();
			edgeWeights.add(normalization * dstNormalizationFactor);
		}
	}

	/**
	* Aggregate the normalized features in the most recent layer for the neighbors of this vertex
	* @return ArrayList<Double>
	*/
	public ArrayList<Double> getNormalizedNeighborFeatures() {
		// Initialize an the size of the features in the current layer
		// Set to zero
		ArrayList<Double> result = new ArrayList<Double>(getCurrentFeatures().size());
		for (int i = 0; i < getCurrentFeatures().size(); i++) result.add(0.0);
		
		// Sum the vectors in result weighted by the normalization
		// on the corresponding edge
		for (int i = 0; i < edgeIndices.size(); i++) {
			ArrayList<Double> temp = VectorFunctions.multiplyByScalar(edgeWeights.get(i), graph.get(edgeIndices.get(i)).getCurrentFeatures()); 
			result = VectorFunctions.sumVectors(result, temp);
		}

		return result;
	}


	// utility
	public String toString() {
		String result = "(" + Integer.toString(vertexId) + "):";
		for (int i = 0; i < edgeIndices.size(); i++) {
			result += " " + Integer.toString(edgeIndices.get(i));
			if (edgeIndices.size() == edgeWeights.size()) {
				result += ", " + Double.toString(edgeWeights.get(i));
			}
			result += ";";
		}

		return result;
	}

	public void printActivations(int layer) {
		for (double d: getFeatures(layer)) System.out.printf(Double.toString(d) + " | ");
		System.out.println();
	}
}








