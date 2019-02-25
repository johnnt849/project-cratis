package engine;

import java.util.ArrayList;

import data.Vertex;
import engine.GCN;
import engine.Preprocess;

public class Main {
	public static void main(String[] args) {
		// Preprocessing
		String featFile = args[0];
		String edgeFile = args[1];
		Preprocess prep = new Preprocess();

		ArrayList<Vertex> graph = prep.run(featFile, edgeFile);

		GCN cn = new GCN(graph);

		int numIters = 1;
		cn.run(graph, numIters);
	}
}
