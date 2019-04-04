
# Local diffusion using Python

# imports
import numpy as np 

# constants
MATRIX_SIZE = 8 # size of square grid
BLOCK_SIZE = 2 # number of blocks
PROBABILITY = 0.5 # probability of diffusion

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
			__global__ void diffuse(float* grid, float* new_grid, float prob)
			{{

				unsigned int grid_size = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;			// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;			// row element of index
				unsigned int thread_id = y * m_size + x; 						// thread index in array

				unsigned int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

				if ((grid[thread_id] == 1) && !(edge)) {{
					unsigned int above = thread_id - m_size;
					unsigned int above_left = thread_id - m_size - 1;
					unsigned int above_right = thread_id - m_size + 1;
					unsigned int below = thread_id + m_size;
					unsigned int below_left = thread_id + m_size - 1;
					unsigned int below_right = thread_id + m_size + 1;
					unsigned int left = thread_id - 1;
					unsigned int right = thread_id + 1;

					new_grid[above] = 1;
					new_grid[above_left] = 1;
					new_grid[above_right] = 1;
					new_grid[below] = 1;
					new_grid[below_left] = 1;
					new_grid[below_right] = 1;
					new_grid[left] = 1;
					new_grid[right] = 1;
				}}
			}}
		"""

		# Transfer CPU memory to GPU memory
		self.grid_gpu = gpuarray.to_gpu(self.grid)
		self.new_grid = gpuarray.empty((self.size, self.size), np.float32)

		self.kernel = self.kernel_code.format(self.size)

		# Compile kernel code
		self.mod = SourceModule(self.kernel)

		# Get kernel function from compiled module
		self.diffusion = self.mod.get_function('diffuse')

	def run(self):
		print('Starting grid: ', self.grid)
		self.diffusion(
			# input
			self.grid_gpu,
			# output
			self.new_grid,
			# grid of n_blocks x n_blocks
			grid = (self.n_blocks, self.n_blocks, 1),
			# block 0f n_threads x n_threads
			block = (self.n_threads, self.n_threads, 1)
			)
		print('Final grid: ', self.new_grid)

if __name__ == '__main__':
	Diffusion(MATRIX_SIZE, BLOCK_SIZE, PROBABILITY)