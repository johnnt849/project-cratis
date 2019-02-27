package engine;

import java.util.ArrayList;

import util.Matrix;
import util.MatrixImpl;
import util.Vector;
import util.VectorFunctions;
import util.VectorImpl;

public class Reducer extends Thread {
	ArrayList< ArrayList<Vector> > buckets;
	int bucketNum;
	Matrix resultMatrix;

	public Reducer(ArrayList< ArrayList<Vector> > bkt, int id, Matrix res) {
		buckets = bkt;
		bucketNum = id;
		resultMatrix = res;
	}

	public void run() {
		Vector result = new VectorImpl();
		for (int i = 0; i < buckets.get(bucketNum).get(0).size(); i++) result.add(0.0);
		
		for (Vector v: buckets.get(bucketNum)) result = sumVectors(result, v);

		resultMatrix.set(bucketNum, result);
	}

	public Vector sumVectors(Vector v1, Vector v2) {
		Vector res = new VectorImpl(v1.size());
		for (int i = 0; i < v1.size(); i++) res.add(v1.get(i) + v2.get(i));

		return res;
	}
}
