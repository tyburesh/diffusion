
# Local diffusion using Python

# imports
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule

# constants
MATRIX_SIZE = 8 # size of square grid
BLOCK_SIZE = 2 # block dimensions
PROBABILITY = 0.75 # probability of diffusion
N_ITERS = 5 # number of iterations

class Diffusion:
	def __init__(self, matrix, block, probability):
		self.size = matrix
		self.n_blocks = matrix // block
		self.n_threads = block
		self.prob = probability
		self.initialize_grid()
		self.initialize_kernel()
		self.run()

	def initialize_grid(self):
		self.grid = np.zeros((self.size, self.size)).astype(np.float32)
		self.grid[self.size // 2][self.size // 2] = 1 # seed is in the center of the matrix

	def initialize_kernel(self):
		self.kernel_code = """

			#include <stdlib.h>
			#include <math.h>

			// Ignore edge rows and columns
			__global__ void diffuse(float* grid, float* new_grid, float* randoms, int* x_coords, int* y_coords)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;				// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;				// row element of index
				unsigned int thread_id = y * grid_size + x; 						// thread index in array

				if (grid[thread_id] == 1) {{
					new_grid[thread_id] = 1;										// current cell
					if (randoms[thread_id] < prob) {{

						// row and col before distance decay
						unsigned int random_x = x_coords[thread_id];
						unsigned int random_y = y_coords[thread_id];

						float diff = prob - randoms[thread_id];

						// distance decay occuring in x and y directions
						// amount of decay dictated by random coordinate, diffusion threshold, and random value
						float decay_x = floor(abs(((float)random_x - x) / prob * diff));
						float decay_y = floor(abs(((float)random_y - y) / prob * diff));

						// apply decay in appropriate direction
						unsigned int spread_x = random_x;
						if (random_x > x) {{
							spread_x -= decay_x;
						}}
						else if (random_x < x) {{
							spread_x += decay_x;
						}}

						// apply decay in appropriate direction
						unsigned int spread_y = random_y;
						if (random_y > y) {{
							spread_y -= decay_y;
						}}
						else if (random_y < y) {{
							spread_y += decay_y;
						}}

						/*
						printf("Initial y: %u\\t"
							"Inintial x: %u\\t"
							"Random y: %u\\t"
							"Random x: %u\\t"
							"Y decay: %f\\t"
							"Decay x: %f\\t"
							"New y: %u\\t"
							"New x: %u\\n",
							y, x, random_y, random_x, decay_y, decay_x, spread_y, spread_x);
						*/

						unsigned int spread_index = spread_y * grid_size + spread_x;
						new_grid[spread_index] = 1;
					}}
				}}
			}}
		"""

		# Transfer CPU memory to GPU memory
		self.grid_gpu = gpuarray.to_gpu(self.grid)
		self.new_grid = gpuarray.empty((self.size, self.size), np.float32)

		self.kernel = self.kernel_code.format(self.size, self.prob)

		# Compile kernel code
		self.mod = SourceModule(self.kernel)

		# Get kernel function from compiled module
		self.diffusion = self.mod.get_function('diffuse')

		# random numbers indicating probabilty of diffusion to a given cell
		self.randoms = curandom.rand((self.size, self.size))
		self.random_x_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)
		self.random_y_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)

	def run(self):
		print('Starting grid: \n', self.grid_gpu)
		i = 0
		while i < N_ITERS:
			self.diffusion(
				# input
				self.grid_gpu,
				# output
				self.new_grid,
				# random numbers
				self.randoms,
				# x coordinates
				self.random_x_coordinates,
				# y coordinates
				self.random_y_coordinates,
				# grid of n_blocks x n_blocks
				grid = (self.n_blocks, self.n_blocks, 1),
				# block 0f n_threads x n_threads
				block = (self.n_threads, self.n_threads, 1),
				)
			self.grid_gpu, self.new_grid = self.new_grid, self.grid_gpu
			self.randoms = curandom.rand((self.size, self.size))
			self.random_x_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)
			self.random_y_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)
			i += 1
			print('\nGrid after iteration {}: \n{}'.format(i, self.grid_gpu))
		print('\nFinal grid: \n', self.grid_gpu)

if __name__ == '__main__':
	Diffusion(MATRIX_SIZE, BLOCK_SIZE, PROBABILITY)
