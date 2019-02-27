package util;

import java.util.ArrayList;

import util.Vector;

public class MatrixImpl extends ArrayList<Vector> implements Matrix {
	public MatrixImpl(int size) {
		super(size);
	}

	public MatrixImpl() {}

	public void shape() {
		System.out.println("(" + Integer.toString(super.size()) + "," + Integer.toString(super.get(0).size()) + ")");
	}
}
