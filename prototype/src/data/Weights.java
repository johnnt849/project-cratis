package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;

import util.Matrix;
import util.MatrixImpl;
import util.Vector;
import util.VectorImpl;

public class Weights {
	int rows;
	int cols;
	Matrix wm;
	Matrix wmT;

	boolean updated;

	public Weights(int x, int y) {
		// because we are not using matrices we need to 
		// switch col and row sizes
		// switched so they don't have to be dealt with later
		rows = y;
		cols = x;
		updated = false;

		// Transpose the initial matrix to work for forward propagation
		wm = new MatrixImpl(rows);
		for (int i = 0; i < rows; i++) {
			Vector c = new VectorImpl(cols);
			for (int j = 0; j < cols; j++) {
				c.add(Math.random()*3 - 1.5);	// initialize -1.5 < x < 1.5
			}
			wm.add(c);
		}

		// Create a trasnpose of the weight matrix for back propagation
		wmT = new MatrixImpl(cols);
		for (int i = 0; i < cols; i++) {
			Vector r = new VectorImpl(rows);
			for (int j = 0; j < rows; j++) {
				r.add(wm.get(j).get(i));
			}
			wmT.add(r);
		}
	}

	/**
	* Read a file representing the weight matrix 
	* (Used mostly for comparison)
	*/
	public Weights(String weightFile) {
		try {
			FileReader fr = new FileReader(weightFile);
			BufferedReader br = new BufferedReader(fr);

			wmT = new MatrixImpl();
			String line;
			while ((line = br.readLine()) != null) {
				Vector v = new VectorImpl();
				String[] parts = line.split(" ");
				for (String s: parts) {
					v.add(Double.parseDouble(s));
				}
				wmT.add(v);
			}
	
			br.close();
		} catch(IOException e) {
			System.err.println("File not found");
		}

		rows = wmT.get(0).size();
		cols = wmT.size(); 

		wm = new MatrixImpl(rows);
		for (int i = 0; i < rows; i++) {
			Vector c = new VectorImpl(cols);
			for (int j = 0; j < cols; j++) {
				c.add(wmT.get(j).get(i));
			}
			wm.add(c);
		}

		updated = false;
	}

	// getters
	public Vector get(int ind) { return wm.get(ind); }
	public Vector getT(int ind) { return wmT.get(ind); }
	public int getNumRows() { return rows; }
	public int getNumCols() { return cols; }
	public String shape() { return "(" + Integer.toString(cols) + "," + Integer.toString(rows) + ")"; }

	// setters
	public void set(int ind, Vector v) { wm.set(ind, v); updated = true; }

	public void createTranspose() {
		if (!updated) return;		// don't create transpose if original and transpose already match

		wmT = new MatrixImpl(cols);
		for (int i = 0; i < cols; i++) {
			Vector r = new VectorImpl(rows);
			for (int j = 0; j < rows; j++) {
				r.add(wm.get(j).get(i));
			}
			wmT.add(r);
		}

		updated = false;
	}
	

	public String toString() {
		String result = "";
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result += Double.toString(wm.get(i).get(j)) + " ";
			}
			result += "\n";
		}

		return result;
	}

	public String toStringTranspose() {
		String result = "";
		for (int i = 0; i < cols; i++) {
			for (int j = 0; j < rows; j++) {
				result += Double.toString(wmT.get(i).get(j)) + " ";
			}
			result += "\n";
		}

		return result;
	}
}
