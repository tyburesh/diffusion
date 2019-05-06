
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
BLOCK_DIMS = 2 # block dimensions
GRID_DIMS = (MATRIX_SIZE + BLOCK_DIMS - 1) // BLOCK_DIMS # grid dimensions
P_LOCAL = 1.0 # probability of local diffusion
P_NON_LOCAL = 0.50 # probability of non-local diffusion
N_ITERS = 1 # number of iterations

class Diffusion:
	def __init__(self, matrix, block, grid, local, non_local):
		self.size = matrix
		self.block_dims = block
		self.grid_dims = grid
		self.p_local = local
		self.p_non_local = non_local
		self.initialize_grid()
		self.initialize_gpu_memory()
		self.initialize_kernel()
		self.initialize_randoms()
		self.run()	# unnecessary for FOREST implementation

	# Create MATRIX_SIZE x MATRIX_SIZE numpy array and initialize seed
	def initialize_grid(self):
		self.grid = np.zeros((self.size, self.size)).astype(np.float32)
		#self.grid[self.size // 2][self.size // 2] = 1 # seed is in the center of the matrix
		self.grid[0][0] = 1
		self.grid[4][4] = 1

	# Transfer CPU memory to GPU memory
	def initialize_gpu_memory(self):
		self.grid_a = gpuarray.to_gpu(self.grid)
		self.grid_b = gpuarray.empty((self.size, self.size), np.float32)

	# Create kernel and kernel functions
	def initialize_kernel(self):
		self.kernel_code = """

			#include <curand_kernel.h>
			#include <math.h>

			extern "C" {{

			__global__ void local_diffuse(float* grid_a, float* grid_b, curandState* global_state)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;				// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;				// row element of index

				if (x < grid_size && y < grid_size) {{

					unsigned int thread_id = y * grid_size + x; 					// thread index in array
					float num;

					// edges will be ignored as starting points
					unsigned int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

					if (grid_a[thread_id] == 1) {{
						grid_b[thread_id] = 1;										// current cell
						if (!edge) {{
							curandState local_state = global_state[thread_id];
							num = curand_uniform(&local_state);
							if (num < prob) {{
								grid_b[thread_id - grid_size] = 1;					// above
							}}
							num = curand_uniform(&local_state);
							if (num < prob) {{
								grid_b[thread_id - grid_size - 1] = 1;				// above and left
							}}
							num = curand_uniform(&local_state);
							if (num < prob) {{
								grid_b[thread_id - grid_size + 1] = 1;				// above and right
							}}
							num = curand_uniform(&local_state);
							if (num < prob) {{
								grid_b[thread_id + grid_size] = 1;					// below
							}}
							num = curand_uniform(&local_state);
							if (num < prob) {{
								grid_b[thread_id + grid_size - 1] = 1;				// below and left
							}}
							num = curand_uniform(&local_state);
							if (num < prob) {{
								grid_b[thread_id + grid_size + 1] = 1;				// below and right
							}}
							num = curand_uniform(&local_state);
							if (num < prob) {{
								grid_b[thread_id - 1] = 1;							// left
							}}
							num = curand_uniform(&local_state);
							if (num < prob) {{
								grid_b[thread_id + 1] = 1;							// right
							}}
							global_state[thread_id] = local_state;
						}}
					}}
				}}
			}}

			__global__ void non_local_diffuse(float* grid_a, float* grid_b, curandState* global_state) {{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;				// column index of element
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;				// row element of index

				if (x < grid_size && y < grid_size) {{

					unsigned int thread_id = y * grid_size + x;						// thread index in array
					float num;
					unsigned int x_coord;
					unsigned int y_coord;
					unsigned int spread_index;

					if (grid_a[thread_id] == 1) {{
						grid_b[thread_id] = 1;										// current cell

						// generate random number between (0,1]
						curandState local_state = global_state[thread_id];
						num = curand_uniform(&local_state);

						// non-local diffusion occurs for until a number > prob is randomly generated
						while (num < prob) {{
							x_coord = (int) truncf(curand_uniform(&local_state) * (grid_size - 0.000001));
							y_coord = (int) truncf(curand_uniform(&local_state) * (grid_size - 0.000001));
							spread_index = y_coord * grid_size + x_coord;
							grid_b[spread_index] = 1;
							num = curand_uniform(&local_state);
						}}
						global_state[thread_id] = local_state;						// copy state back to global memory
					}}
				}}
			}}
			}}
		"""

		self.kernel = self.kernel_code.format(self.size, self.p_local, self.size, self.p_non_local)
		self.mod = SourceModule(self.kernel, no_extern_c = True)
		self.local_diffusion = self.mod.get_function('local_diffuse')
		self.non_local_diffusion = self.mod.get_function('non_local_diffuse')

	# Create random number generator to be used during kernel execution
	def initialize_randoms(self):
		self.generator = curandom.XORWOWRandomNumberGenerator()

	# Performs one iteration of local diffusion
	def local(self):
		print('\nGrid before local diffusion: \n', self.grid_a.get())
		self.local_diffusion(
			self.grid_a, self.grid_b, self.generator.state,
			grid = (self.grid_dims, self.grid_dims, 1),
			block = (self.block_dims, self.block_dims, 1),)
		self.grid_a, self.grid_b = self.grid_b, self.grid_a
		print('\nGrid after local diffusion: \n', self.grid_a.get())

	# Performs one iteration of non_local diffusion
	def non_local(self):
		print('\nGrid before non_local diffusion: \n', self.grid_a.get())
		self.non_local_diffusion(
			self.grid_a, self.grid_b, self.generator.state,
			grid = (self.grid_dims, self.grid_dims, 1),
			block = (self.block_dims, self.block_dims, 1),)
		self.grid_a, self.grid_b = self.grid_b, self.grid_a
		print('\nGrid after non_local diffusion: \n', self.grid_a.get())

	# unnecessary for FOREST implementation
	def run(self):
		i = 0
		while i < N_ITERS:
			#self.local()
			self.non_local()
			i += 1
		print('\nFinal Board: ', self.grid_a.get())

if __name__ == '__main__':
	Diffusion(MATRIX_SIZE, BLOCK_DIMS, GRID_DIMS, P_LOCAL, P_NON_LOCAL)

