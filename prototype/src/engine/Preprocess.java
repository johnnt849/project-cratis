package engine;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;

import data.Vertex;

public class Preprocess {
	public ArrayList<Vertex> run(String featureFile, String edgeFile) {
		ArrayList<Vertex> g = new ArrayList<Vertex>();

		try {
			// read and process feature file
			FileReader ff = new FileReader(featureFile);
			BufferedReader fbr = new BufferedReader(ff);

			readFeatures(fbr, g);
			fbr.close();

			// read and process edge file
			FileReader ef = new FileReader(edgeFile);
			BufferedReader ebr = new BufferedReader(ef);

			addEdges(ebr, g);
			ebr.close();
		} catch (IOException e) {
			System.err.println("FILE NOT FOUND");
		}

		for (Vertex v: g) v.setNormalization();
		for (Vertex v: g) v.findEdgeNormalization();

		return g;
	}

	private void readFeatures(BufferedReader fbr, ArrayList<Vertex> g) {
		String line;
		try {
			while ((line = fbr.readLine()) != null) {
				String[] parts = line.split("\t");
				String[] featStrs = parts[1].split(",");
				int src = Integer.parseInt(parts[0]);
				int classification = Integer.parseInt(parts[2]);

				ArrayList<Double> feats = new ArrayList<Double>();
				for (int i = 0; i < featStrs.length; i++) {
					feats.add(Double.parseDouble(featStrs[i]));
				}

				Vertex v = new Vertex(src, g, classification, feats);
				v.insertEdge(src);
				g.add(v);
			}
		} catch (IOException e) {
			System.err.println("SOME IO EXCEPTION");
		}
	}

	private void addEdges(BufferedReader br, ArrayList<Vertex> g) {
		String line;
		Vertex v;
		int src, dst;
		try {
			while ((line = br.readLine()) != null) {
				String[] verts = line.split("\t");
				src = Integer.parseInt(verts[0]);
				dst = Integer.parseInt(verts[1]);

				v = g.get(src);
				v.insertEdge(dst);
			}
		} catch(IOException e) {
			System.out.println("SOME IO EXCEPTION");
		}
	}
}
