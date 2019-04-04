
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
PROBABILITY = 0.5 # probability of diffusion
N_ITERS = 2 # number of iterations

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

			// Ignore edge rows and columns
			// Assuming the matrix is large, the effect of this is small
			__global__ void diffuse(float* grid, float* new_grid, float* randoms)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;			// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;			// row element of index
				unsigned int thread_id = y * grid_size + x; 					// thread index in array

				unsigned int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

				if ((grid[thread_id] == 1) && !(edge)) {{
					new_grid[thread_id] = 1;									// current cell
					if (randoms[thread_id - grid_size] > prob) {{
						new_grid[thread_id - grid_size] = 1;					// above
					}}
					if (randoms[thread_id - grid_size - 1] > prob) {{
						new_grid[thread_id - grid_size - 1] = 1;				// above and left
					}}
					if (randoms[thread_id - grid_size + 1] > prob) {{
						new_grid[thread_id - grid_size + 1] = 1;				// above and right
					}}
					if (randoms[thread_id + grid_size] > prob) {{
						new_grid[thread_id + grid_size] = 1;					// below
					}}
					if (randoms[thread_id + grid_size - 1] > prob) {{
						new_grid[thread_id + grid_size - 1] = 1;				// below and left
					}}
					if (randoms[thread_id + grid_size + 1] > prob) {{
						new_grid[thread_id + grid_size + 1] = 1;				// below and right
					}}
					if (randoms[thread_id - 1] > prob) {{
						new_grid[thread_id - 1] = 1;							// left
					}}
					if (randoms[thread_id + 1] > prob) {{
						new_grid[thread_id + 1] = 1;							// right
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

		self.randoms = curandom.rand((self.size, self.size))

	def run(self):
		print('Starting grid: ', self.grid_gpu)
		i = 0
		while i < N_ITERS:
			self.diffusion(
				# input
				self.grid_gpu,
				# output
				self.new_grid,
				# random floats used for diffusion
				self.randoms,
				# grid of n_blocks x n_blocks
				grid = (self.n_blocks, self.n_blocks, 1),
				# block 0f n_threads x n_threads
				block = (self.n_threads, self.n_threads, 1),
				)
			self.grid_gpu, self.new_grid = self.new_grid, self.grid_gpu
			self.randoms = curandom.rand((self.size, self.size))
			i += 1
		print('Final grid: ', self.grid_gpu)

if __name__ == '__main__':
	Diffusion(MATRIX_SIZE, BLOCK_SIZE, PROBABILITY)