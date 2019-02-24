package data;

import java.lang.Math;
import java.util.ArrayList;

public class Weights {
	int rows;
	int cols;
	ArrayList< ArrayList<Double> > wm;
	ArrayList< ArrayList<Double> > wmT;

	public Weights(int x, int y) {
		// because we are not using matrices we need to 
		// switch col and row sizes
		// switched so they don't have to be dealt with later
		rows = y;
		cols = x;

		// Transpose the initial matrix to work for forward propagation
		wm = new ArrayList< ArrayList<Double> >(rows);
		for (int i = 0; i < rows; i++) {
			ArrayList<Double> c = new ArrayList<Double>(cols);
			for (int j = 0; j < cols; j++) {
				c.add(Math.random()*3 - 1.5);	// initialize -1.5 < x < 1.5
			}
			wm.add(c);
		}

		// Create a trasnpose of the weight matrix for back propagation
		wmT = new ArrayList< ArrayList<Double> >(cols);
		for (int i = 0; i < cols; i++) {
			ArrayList<Double> r = new ArrayList<Double>(rows);
			for (int j = 0; j < rows; j++) {
				r.add(wm.get(j).get(i));
			}
			wmT.add(r);
		}
	}

	public ArrayList<Double> get(int ind) { return wm.get(ind); }
	public ArrayList<Double> getT(int ind) { return wmT.get(ind); }
	public int getNumRows() { return rows; }
	public int getNumCols() { return cols; }
	public String shape() { return "(" + Integer.toString(cols) + "," + Integer.toString(rows) + ")"; }

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
