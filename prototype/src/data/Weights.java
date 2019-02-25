package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
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

	/**
	* Read a file representing the weight matrix 
	* (Used mostly for comparison)
	*/
	public Weights(String weightFile) {
		try {
			FileReader fr = new FileReader(weightFile);
			BufferedReader br = new BufferedReader(fr);

			wmT = new ArrayList< ArrayList<Double> >();
			String line;
			while ((line = br.readLine()) != null) {
				ArrayList<Double> v = new ArrayList<Double>();
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

		wm = new ArrayList< ArrayList<Double> >(rows);
		for (int i = 0; i < rows; i++) {
			ArrayList<Double> c = new ArrayList<Double>(cols);
			for (int j = 0; j < cols; j++) {
				c.add(wmT.get(j).get(i));
			}
			wm.add(c);
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
