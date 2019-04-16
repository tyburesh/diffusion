
# Diffusion using Python
# Uses local and non-local diffusion

# imports
import numpy as np
import sys
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule

# constants
MATRIX_SIZE = 8 # size of square grid
BLOCK_SIZE = 2 # block dimensions
P_LOCAL = 0.1 # probability of local diffusion
P_NON_LOCAL = 0.25 # probability of non-local diffusion
N_ITERS = 1 # number of iterations
N_STREAMS = 4 # data broken into four quadrants
STREAM_SIZE = MATRIX_SIZE * MATRIX_SIZE // N_STREAMS # data elements per stream


class Diffusion:
	def __init__(self, matrix, block, local, non_local):
		self.size = matrix
		self.n_blocks = matrix // block
		self.n_threads = block
		self.p_local = local
		self.p_non_local = non_local
		self.initialize_grid()
		self.initialize_gpu_memory()
		self.initialize_kernel()
		self.run()	# unnecessary for FOREST implementation

	# Create MATRIX_SIZE x MATRIX_SIZE numpy array and initialize seed
	def initialize_grid(self):
		self.grid = np.zeros((self.size, self.size)).astype(np.float32)
		self.grid[self.size // 2][self.size // 2] = 1 # seed is in the center of the matrix

	# Allocate memory on GPU
	def initialize_gpu_memory(self):
		self.grid_a = gpuarray.empty(STREAM_SIZE, np.float32)
		self.grid_b = gpuarray.empty(STREAM_SIZE, np.float32)
	#	self.grid_a = gpuarray.to_gpu(self.grid)
	#	self.grid_b = gpuarray.empty((self.size, self.size), np.float32)

	# Create kernel and kernel functions
	def initialize_kernel(self):
		self.kernel_code = """
			__global__ void local_diffuse(float* grid_a, float* grid_b, float* randoms)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;			// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;			// row element of index
				unsigned int thread_id = y * grid_size + x; 					// thread index in array

				// edges will be ignored as starting points
				unsigned int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

				if (grid_a[thread_id] == 1) {{
					grid_b[thread_id] = 1;										// current cell
					if (!edge) {{
						if (randoms[thread_id - grid_size] < prob) {{
							grid_b[thread_id - grid_size] = 1;					// above
						}}
						if (randoms[thread_id - grid_size - 1] < prob) {{
							grid_b[thread_id - grid_size - 1] = 1;				// above and left
						}}
						if (randoms[thread_id - grid_size + 1] < prob) {{
							grid_b[thread_id - grid_size + 1] = 1;				// above and right
						}}
						if (randoms[thread_id + grid_size] < prob) {{
							grid_b[thread_id + grid_size] = 1;					// below
						}}
						if (randoms[thread_id + grid_size - 1] < prob) {{
							grid_b[thread_id + grid_size - 1] = 1;				// below and left
						}}
						if (randoms[thread_id + grid_size + 1] < prob) {{
							grid_b[thread_id + grid_size + 1] = 1;				// below and right
						}}
						if (randoms[thread_id - 1] < prob) {{
							grid_b[thread_id - 1] = 1;							// left
						}}
						if (randoms[thread_id + 1] < prob) {{
							grid_b[thread_id + 1] = 1;							// right
						}}
					}}
				}}
			}}

			__global__ void non_local_diffuse(float* grid_a, float* grid_b, float* randoms, int* x_coords, int* y_coords)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;			// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;			// row element of index
				unsigned int thread_id = y * grid_size + x; 					// thread index in array

				if (grid_a[thread_id] == 1) {{
					grid_b[thread_id] = 1;									// current cell
					if (randoms[thread_id] < prob) {{
						unsigned int spread_index = y_coords[thread_id] * grid_size + x_coords[thread_id];
						grid_b[spread_index] = 1;
					}}
				}}
			}}
		"""

		self.kernel = self.kernel_code.format(self.size, self.p_local, self.size, self.p_non_local)
		self.mod = SourceModule(self.kernel)
		self.local_diffusion = self.mod.get_function('local_diffuse')
		self.non_local_diffusion = self.mod.get_function('non_local_diffuse')

	# Gererate random values between [0,1)
	# Diffusion will occur / not occur based on these values
	def generate_randoms(self):
		self.randoms = curandom.rand((self.size, self.size))

	# Generate random coordinates between [0, MATRIX_SIZE - 1]
	# Dictate non_local diffusion location (if it occurs)
	def generate_random_coords(self):
		self.random_x_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)
		self.random_y_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)

	# Performs one iteration of local diffusion
	def local(self):
		print('\nGrid before local diffusion: \n', self.grid)
		for i in range(N_STREAMS):

			offset = i * STREAM_SIZE

			# TODO: fix indexing of self.grid
			self.grid_a = gpuarray.to_gpu(self.grid[0:2])
			self.grid_b = gpuarray.to_gpu(self.grid[0:2])

			self.local_diffusion(
				self.grid_a, self.grid_b, self.randoms,
				grid = (self.n_blocks, self.n_blocks, 1),
				block = (self.n_threads, self.n_threads, 1),)
			self.grid_a, self.grid_b = self.grid_b, self.grid_a

			# TODO: fix indexing of self.grid
			self.grid[0:2] = self.grid_a.get()

		print('\nGrid after local diffusion: \n', self.grid)

	# Performs one iteration of non_local diffusion
	def non_local(self):
		print('\nGrid before non_local diffusion: \n', self.grid_b)
		self.non_local_diffusion(
			self.grid_b, self.grid_a, self.randoms, 
			self.random_x_coordinates, self.random_y_coordinates,
			grid = (self.n_blocks, self.n_blocks, 1),
			block = (self.n_threads, self.n_threads, 1),)
		self.grid_a, self.grid_b = self.grid_b, self.grid_a
		print('\nGrid after non_local diffusion: \n', self.grid_a)

	# unnecessary for FOREST implementation
	def run(self):
		i = 0
		while i < N_ITERS:
			self.generate_randoms()
			self.local()
			#self.generate_random_coords()
			#self.non_local()
			i += 1
		#print('Final Board: ', grid_b.get())

if __name__ == '__main__':
	Diffusion(MATRIX_SIZE, BLOCK_SIZE, P_LOCAL, P_NON_LOCAL)

