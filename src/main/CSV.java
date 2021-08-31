/**
 * MIT License
 * 
 * Copyright (c) 2018 Justin Kunimune
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * a class for reading the weird input text files that go with this.
 * 
 * @author Justin Kunimune
 */
public class CSV {
	
	/**
	 * read a simple CSV file, with any standard line break character, and return its contents
	 * as a double matrix. file must end with a line break, and elements must be parsable as
	 * doubles. whitespace adjacent to delimiters will be stripped.
	 * @param file the CSV file to open
	 * @param delimiter the delimiting character, usually ',', sometimes '\t', occasionally '|'
	 * @return I think the return value is pretty self-explanatory.
	 * @throws IOException if file cannot be found or permission is denied
	 * @throws NumberFormatException if elements are not parsable as doubles
	 */
	public static double[][] read(File file, char delimiter)
			throws NumberFormatException, IOException {
		return read(file, delimiter, 0);
	}
	
	/**
	 * read a simple CSV file, with any standard line break character, and return its contents
	 * as a double matrix. file must end with a line break, and elements must be parsable as
	 * doubles. whitespace adjacent to delimiters will be stripped.
	 * @param file the CSV file to open
	 * @param delimiter the delimiting character, usually ',', sometimes '\t', occasionally '|'
	 * @param headerRows the number of initial rows to skip
	 * @return I think the return value is pretty self-explanatory.
	 * @throws IOException if file cannot be found or permission is denied
	 * @throws NumberFormatException if elements are not parsable as doubles
	 */
	public static double[][] read(File file, char delimiter, int headerRows)
			throws NumberFormatException, IOException {
		List<double[]> list;
		try (BufferedReader in = new BufferedReader(new FileReader(file))) {
			list = new ArrayList<double[]>();
			String line;
			for (int i = 0; i < headerRows; i ++)
				in.readLine();
			while ((line = in.readLine()) != null) {
				line = line.trim();
				if (line.isEmpty())
					break;
				String[] elements = line.split("\\s*" + delimiter + "\\s*");
				double[] row = new double[elements.length];
				for (int j = 0; j < elements.length; j++)
					row[j] = Double.parseDouble(elements[j]);
				list.add(row);
			}
		}
		return list.toArray(new double[0][]);
	}
	
	/**
	 * read a CSV file where there is only one column, bypassing the need for a multi-
	 * dimensional array. values will be separated by line breaks and adjacent whitespace
	 * alone, and will be returned in a 1D array.
	 * @param file the CSV file to open
	 * @return 1D array of values from the list
	 * @throws IOException if file cannot be found or permission is denied
	 * @throws NumberFormatException if elements are not parsable as doubles
	 */
	public static double[] readColumn(File file)
			throws NumberFormatException, IOException {
		return readColumn(file, '\n', 0);
	}
	
	/**
	 * read a CSV file and return a single column as a 1D array.
	 * @param file the CSV file to open
	 * @param delimiter the delimiting character, usually ',', sometimes '\t', occasionally '|'
	 * @param j index of the desired column
	 * @return 1D array of values in column j
	 * @throws IOException if file cannot be found or permission is denied
	 * @throws NumberFormatException if elements are not parsable as doubles
	 */
	public static double[] readColumn(File file, char delimiter, int j)
			throws NumberFormatException, IOException {
		double[][] table = read(file, delimiter);
		double[] out = new double[table.length];
		for (int i = 0; i < table.length; i ++)
			out[i] = table[i][j];
		return out;
	}
	
	/**
	 * read a COSY-generated file and return the coefficients as a double matrix along with the
	 * exponents as an int matrix.
	 * @param file the COSY file to open
	 * @param maxOrder the desired order of the polynomial
	 * @return yep
	 * @throws IOException if file cannot be found or permission is denied
	 * @throws NumberFormatException if elements are not parsable as doubles
	 */
	public static COSYMapping readCosyCoefficients(File file, int maxOrder)
			throws NumberFormatException, IOException {
		List<double[]> coefList;
		List<int[]> expList;
		try (BufferedReader in = new BufferedReader(new FileReader(file))) { // start by reading it like a CSV
			coefList = new ArrayList<double[]>();
			expList = new ArrayList<int[]>();
			String line;
			while ((line = in.readLine()) != null) {
				line = line.substring(1); // remove this single stupid space in front of the numbers
				if (line.trim().isEmpty())
					break;
				if (line.substring(line.length()/14*14).contains(Integer.toString(maxOrder + 1))) // check for when we see an exponent over the highest order
					break;
				double[] row = new double[line.length()/14];
				for (int i = 0; i < line.length()/14; i ++)
					row[i] = Double.parseDouble(line.substring(14*i, 14*(i+1)));
				coefList.add(row);
				int[] bloc = new int[line.length() - row.length*14-1];
				for (int i = 0; i < bloc.length; i ++)
					bloc[i] = Integer.parseInt(String.valueOf(line.charAt(row.length*14+1 + i)));
				expList.add(bloc);
			}
		}
		return new COSYMapping(coefList.toArray(new double[0][]), expList.toArray(new int[0][]));
	}
	
	/**
	 * save the given matrix as a simple CSV file, using the given delimiter character.
	 * @param data the numbers to be written
	 * @param file the file at which to save
	 * @param delimiter the delimiter character, usually ','
	 * @throws IOException if the file cannot be found or permission is denied
	 */
	public static void write(double[][] data, File file, char delimiter)
			throws IOException {
		write(data, file, delimiter, null);
	}
	
	/**
	 * save the given matrix as a simple CSV file, using the given delimiter character.
	 * @param data the numbers to be written
	 * @param file the file at which to save
	 * @param delimiter the delimiter character, usually ','
	 * @param header the list of strings to put on top
	 * @throws IOException if the file cannot be found or permission is denied
	 */
	public static void write(double[][] data, File file, char delimiter, String[] header)
			throws IOException {
		try (BufferedWriter out = new BufferedWriter(new FileWriter(file))) {
			if (header != null) {
				for (int j = 0; j < header.length; j++) {
					out.append(header[j]);
					if (j < header.length - 1)
						out.append(delimiter);
					else
						out.newLine();
				}
			}
			for (double[] datum: data) {
				for (int j = 0; j < datum.length; j++) {
					out.append(Double.toString(datum[j]));
					if (j < datum.length - 1)
						out.append(delimiter);
					else
						out.newLine();
				}
			}
		}
	}
	
	/**
	 * save the given array as a column of newline-separated number strings.
	 * @param data the numbers to write, in 1D form
	 * @param file the file at which to save
	 * @throws IOException if the file cannot be found or permission is denied
	 */
	public static void writeColumn(double[] data, File file) throws IOException {
		double[][] columnVector = new double[data.length][1];
		for (int i = 0; i < data.length; i ++)
			columnVector[i][0] = data[i];
		write(columnVector, file, '\n');
	}
	
	
	public static class COSYMapping {
		public final double[][] coefficients;
		public final int[][] exponents;
		public COSYMapping(double[][] coefficients, int[][] exponents) {
			this.coefficients = coefficients;
			this.exponents = exponents;
		}
	}
	
}
