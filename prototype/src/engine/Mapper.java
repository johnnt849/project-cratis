package engine;

import java.util.List;
import java.util.ArrayList;

import util.Matrix;
import util.MatrixImpl;
import util.Vector;
import util.VectorImpl;

public class Mapper extends Thread {
	ArrayList<ArrayList<Vector>> buckets;
	List<Vector> actChunks;
	List<Vector> deltChunks;

	public Mapper(ArrayList<ArrayList<Vector>> bkt, List<Vector> ac, List<Vector> dc) { 
		buckets = bkt;
		actChunks = ac;
		deltChunks = dc;
	}

	public void run() {
		int pos = 0;
		for (int v = 0; v < actChunks.size(); v++) {
			for (double i: actChunks.get(v)) {
				Vector factors = new VectorImpl(deltChunks.size());
				for (double j: deltChunks.get(v)) {
					factors.add(i * j);
				}

				synchronized(buckets) {
					addFactors(factors, pos);
				}
				pos++;
			}

			pos = 0;
		}
	}

	public synchronized void addFactors(Vector factors, int pos) {
		buckets.get(pos).add(factors);
	}
}
