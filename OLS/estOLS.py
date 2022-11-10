import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(prog = "estOLS.py",
								 description="Perform an ordinary least squares regression. Use with --help for more information.")
parser.add_argument("-x",
					metavar="XMAT",					
                    help="The n*m data matrix. Should be a space delimited file, with each row on a new line, and no trailing spaces. Delimiter may be overridden using the \"-d\" option.")

parser.add_argument("-y",
					metavar="YVEC",
                    help="The n*1 vector of observations. Should be a plain text file, with each value on a new line, and no trailing spaces or other characters.")

parser.add_argument("-o",
					metavar="OUTPUT_LOCATION",
                    help="File to write the final output. If not present, will write to stdout.")

parser.add_argument("-d",
					metavar="DELIMITER",
					type=str,
					default=" ",
                    help="Delimiter between values in input files. Default is single space.")

parser.add_argument("--skiprows",
					metavar="SKIP",
					type=int,
					default=0,
                    help="Number of rows to skip at the beginning of the file. Useful is the file is in CSV format with a header.")

parser.add_argument("--benchmark",
					action = "store_true",
                    help="Run a benchmark using random synthetic data.")

# QUESTION: Ask about whether or not this is needed.
# parser.add_argument("--add-constant-term",
# 					action = "store_true",
#                     help="Whether or not to add a column to the regression to represent a constant term.")

# Parse args.
args = parser.parse_args()

# Assign args to variables.
x_mat_file = args.x
y_vec_file = args.y
output_file = args.o
delimiter = args.d
skip_rows = args.skiprows
benchmark = args.benchmark

# Check argument sanity
if ((not benchmark) and (x_mat_file == None or y_vec_file == None)):
	print("Program must be run with -x and -y arguments when not in benchmark mode!")
	exit()

def load_data(x_mat_file, y_vec_file):
	# Load the data from files into variables.
	print("Loading data...")
	x_mat = None
	y_vec = None
	try:
		x_mat = np.loadtxt(x_mat_file, delimiter=delimiter, skiprows=skip_rows)
	except FileNotFoundError as ex:
		print("Could not find XMAT file!")
		print("Exception details: {ex}".format(ex=ex))
		exit()	
	except Exception as ex:
		print("Could not read read XMAT file as matrix! Perhaps there is an issue with the formatting?")
		print("Exception details: {ex}".format(ex=ex))
		exit()

	try:
		y_vec = np.loadtxt(y_vec_file, delimiter=delimiter, skiprows=skip_rows)
	except FileNotFoundError as ex:
		print("Could not find YVEC file!")
		print("Exception details: {ex}".format(ex=ex))
		exit()	
	except Exception as ex:
		print("Could not read read YVEC file as vector! Perhaps there is an issue with the formatting?")
		print("Exception details: {ex}".format(ex=ex))
		exit()

	print("Data loaded.")
	return x_mat, y_vec

def run_regression(x_matrix, y_vector):
	# Runs the regression
	regression = np.linalg.lstsq(x_matrix, y_vector, rcond=None)[0]
	return regression

def benchmark_regression(n_dimensions, m_dimensions):
	# Prints out a debugging report
	print("N M DATA_GENERATION_TIME REGRESSION_TIME")
	for n_size in n_dimensions:
		for m_size in m_dimensions:
			# Generate the random data, capturing timing
			data_gen_start = time.time()
			x_random_mat = np.random.rand(n_size, m_size)
			y_random_vec = np.random.rand(n_size, 1)
			data_gen_end = time.time()
			data_gen_elapsed = data_gen_end - data_gen_start

			# Perform the regression, capturing timing
			regression_start = time.time()
			run_regression(x_random_mat, y_random_vec)
			regression_end = time.time()
			regression_elapsed = regression_end - regression_start

			# Print results
			print(f"{n_size} {m_size} {data_gen_elapsed} {regression_elapsed}")
	
if benchmark:
	# 100000x10000 overflows laptop RAM; Limit to 50000x5000
	benchmark_regression([5000,10000,20000,50000], [500,1000,2000,5000])
else:
	x_mat, y_vec = load_data(x_mat_file, y_vec_file)
	regression = run_regression(x_mat, y_vec)
	print("Saving output to {output}...".format(output=output_file))
	np.savetxt(output_file, regression)
