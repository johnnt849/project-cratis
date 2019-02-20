package data;

import java.util.ArrayList;

import data.Vertex;

public class Graph {
	ArrayList<Vertex> vertices;

	public Graph() {
		vertices = new ArrayList<Vertex>();
	}

	public void addVertex(Vertex v) { vertices.add(v); }
	public int getNumVertices() { return vertices.size(); }
	public int getNumEdges() {
		int total = 0;
		for (Vertex v: vertices) total += v.getDegree();
		return total;
	}
	public Vertex getVertex(int ind) { return vertices.get(ind); }

	// Preprocessing
	public void normalizeEdges() {
		// add normalization to each vertex
		for (Vertex v: vertices) {
			v.setNormalization();
		}

		// normalize each edge in the graph
		for (Vertex v: vertices) {
			v.findEdgeNormalization();
		}
	}

	// Forward propagation
	public void aggregateFeatures() {
		for (Vertex v: vertices) {
			
		}
	}

	public String toString() {
		String res = "";
		for (Vertex v: vertices) {
			res += v.toString() + "\n";
		}

		return res += "\n";
	}
}
