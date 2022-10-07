/*
 * MIT License
 *
 * Copyright (c) 2022 Justin Kunimune
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

import java.io.IOException;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class Logging {

	public static void configureLogger(Logger logger, String name) throws UnsupportedEncodingException {
		System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));

		logger.getParent().getHandlers()[0].setFormatter(newFormatter("%1$ta %1$tH:%1$tM:%1tS | %2$s | %3$s%4$s%n"));
		logger.getParent().getHandlers()[0].setEncoding("UTF-8");
		try {
			String filename = String.format("results/log-%s.log", name);
			FileHandler handler = new FileHandler(filename, true);
			handler.setFormatter(newFormatter("%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS | %2$s | %3$s%4$s%n"));
			logger.addHandler(handler);
			System.out.println("logging in　to `"+filename+"`");
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	private static Formatter newFormatter(String format) {
		return new SimpleFormatter() {
			public String format(LogRecord record) {
				return String.format(format,
				                     record.getMillis(),
				                     record.getLevel(),
				                     record.getMessage(),
				                     (record.getThrown() != null) ? record.getThrown() : "");
			}
		};
	}
}
