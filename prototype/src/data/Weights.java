package data;

import java.lang.Math;
import java.util.ArrayList;

public class Weights {
	int rows;
	int cols;
	ArrayList< ArrayList<Double> > wm;

	public Weights(int x, int y) {
		// because we are not using matrices we need to 
		// switch col and row sizes
		// switched so they don't have to be dealt with later
		rows = y;
		cols = x;

		wm = new ArrayList< ArrayList<Double> >(rows);
		for (int i = 0; i < rows; i++) {
			ArrayList<Double> c = new ArrayList<Double>(cols);
			for (int j = 0; j < cols; j++) {
				c.add(Math.random()*3 - 1.5);	// initialize -1.5 < x < 1.5
			}
			wm.add(c);
		}
	}

	public ArrayList<Double> get(int ind) { return wm.get(ind); }
	public int getNumRows() { return rows; }
	public int getNumCols() { return cols; }
	public String shape() { return "(" + Integer.toString(rows) + "," + Integer.toString(cols) + ")"; }

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
}
