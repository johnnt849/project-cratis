package data;

import java.util.ArrayList;

public class Deltas {
	ArrayList< ArrayList<Double> > deltas;
	int layer;

	public Deltas(int lyr) {
		layer = lyr;
		deltas = new ArrayList< ArrayList<Double> >();
	}

	public int getLayer() { return layer; }
	public ArrayList<Double> getDelta(int ind) { return deltas.get(ind); }
	public String shape() { 
		String shape = "(" + Integer.toString(deltas.size()) + ",";
		if (!deltas.isEmpty()) shape += Integer.toString(deltas.get(0).size());

		return shape += ")";
	}

	public void addDelta(ArrayList<Double> delta) { deltas.add(delta); }
}
