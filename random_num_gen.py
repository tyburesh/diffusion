
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
N_STREAMS = 4 # data broken into four quadrants
STREAM_SIZE = MATRIX_SIZE * MATRIX_SIZE // N_STREAMS # data elements per stream


class Diffusion:
	def __init__(self, matrix, block, grid, local, non_local):
		self.size = matrix
		self.block_dims = block
		self.grid_dims = grid
		self.p_local = local
		self.p_non_local = non_local
		#self.initialize_grid()
		#self.initialize_gpu_memory()
		#self.initialize_streams()
		self.initialize_kernel()
		#self.run()	# unnecessary for FOREST implementation

	# Create MATRIX_SIZE x MATRIX_SIZE numpy array and initialize seed
	def initialize_grid(self):
		self.grid = np.zeros((self.size, self.size)).astype(np.float32)
		self.grid[1][1] = 1 # seed is in the center of the matrix

	# Allocate memory on GPU
	def initialize_gpu_memory(self):
		self.gpu_grid_a = drv.mem_alloc(2500000000)
		self.gpu_grid_b = drv.mem_alloc(2500000000)
		self.randoms = drv.mem_alloc(2500000000)

	# Create kernel and kernel functions
	def initialize_kernel(self):
		self.kernel_code = """

			#include <curand_kernel.h>
			#include <math.h>

			extern "C" {{

			__global__ void local_diffuse(float* grid_a, float* grid_b, float* randoms)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;				// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;				// row element of index

				if (x < grid_size && y < grid_size) {{

					unsigned int thread_id = y * grid_size + x; 					// thread index in array
					printf("Thread ID = %u\\n", thread_id);

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
			}}

			__global__ void non_local_diffuse(float* grid_a, float* grid_b, float* randoms, int* x_coords, int* y_coords)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;				// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;				// row element of index

				if (x < grid_size && y < grid_size) {{

					unsigned int thread_id = y * grid_size + x; 					// thread index in array

					if (grid_a[thread_id] == 1) {{
						grid_b[thread_id] = 1;										// current cell
						if (randoms[thread_id] < prob) {{
							unsigned int spread_index = y_coords[thread_id] * grid_size + x_coords[thread_id];
							grid_b[spread_index] = 1;
						}}
					}}
				}}
			}}

			__global__ void nld(float* grid_a, float* grid_b, curandState* global_state) {{

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

						// non-local diffusion occurs until diffusion threshold is passed
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

			__global__ void random_nums(curandState *global_state) {{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
				float num;
				float num2;

				if (x < grid_size && y < grid_size) {{
					unsigned int thread_id = y * grid_size + x;
					curandState local_state = global_state[thread_id];
	       			num = curand_uniform(&local_state);
	       			num2 = curand_uniform(&local_state);
	       			while (num < prob) {{
	       				printf("Id = %u\\tNum = %f\\tNum2 = %f\\n", thread_id, num, num2);
	       				num = curand_uniform(&local_state);
	       				num2 = curand_uniform(&local_state);
	       			}}
	       			global_state[thread_id] = local_state;
				}}
			}}
			}}
		"""

		self.kernel = self.kernel_code.format(self.size, self.p_local, self.size, self.p_non_local, self.size, self.p_non_local, self.size, self.p_non_local)
		self.mod = SourceModule(self.kernel, no_extern_c=True)
		self.local_diffusion = self.mod.get_function('local_diffuse')
		self.non_local_diffusion = self.mod.get_function('non_local_diffuse')
		self.func = self.mod.get_function('random_nums')
		self.generator = curandom.XORWOWRandomNumberGenerator()
		#self.randoms = self.generator.gen_uniform((8,), np.float32)
		self.func(self.generator.state, grid = (8,8,1), block = (8,8,1))

	def initialize_streams(self):
		self.streams = []
		self.events = []
		self.reference = drv.Event()
		for k in range(N_STREAMS):
			self.streams.append(drv.Stream())
			self.events.append({
				'local_kernel_begin':drv.Event(),
				'local_kernel_end':drv.Event(),
				'non_local_kernel_begin':drv.Event(),
				'non_local_kernel_end':drv.Event()
			})

	# Gererate random values between [0,1)
	# Diffusion will occur / not occur based on these values
	def generate_randoms(self):
		pass
		#self.randoms = curandom.rand((self.size, self.size))

	# Generate random coordinates between [0, MATRIX_SIZE - 1]
	# Dictate non_local diffusion location (if it occurs)
	def generate_random_coords(self):
		pass
		#self.random_x_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)
		#self.random_y_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)

	# Performs one iteration of local diffusion
	def local(self):
		print('\nGrid before local diffusion: \n', self.grid)

		for i in range(N_STREAMS):

			start_index = i * (STREAM_SIZE // MATRIX_SIZE)
			end_index = start_index + (STREAM_SIZE // MATRIX_SIZE)

			# Need to figure out indexing of self.grid
			# Pass in quadrants rather than rows
			# Generate random numbers to be equal to the size of the quadrant
			if i == 0: # top left
				drv.memcpy_htod(self.gpu_grid_a, np.array([self.grid[i][:2] for i in range(2)]))
			elif i == 1: # top right
				drv.memcpy_htod(self.gpu_grid_a, np.array([self.grid[i][2:] for i in range(2)]))
			elif i == 2: # bottom left
				drv.memcpy_htod(self.gpu_grid_a, np.array([self.grid[i][:2] for i in range(2,4)]))
			elif i == 3: # bottom right
				drv.memcpy_htod(self.gpu_grid_a, np.array([self.grid[i][2:] for i in range(2,4)]))

			self.events[i]['local_kernel_begin'].record(self.streams[i])
			self.local_diffusion(
				self.gpu_grid_a, self.gpu_grid_b, self.randoms,
				grid = (self.grid_dims, self.grid_dims, 1),
				block = (self.block_dims, self.block_dims, 1),
				stream = self.streams[i],)
			self.events[i]['local_kernel_end'].record(self.streams[i])
			self.gpu_grid_a, self.gpu_grid_b = self.gpu_grid_b, self.gpu_grid_a

			drv.memcpy_dtoh(self.grid[start_index:end_index], self.gpu_grid_a)

		print('\nGrid after local diffusion: \n', self.grid)

	# Performs one iteration of non_local diffusion
	def non_local(self):
		print('\nGrid before non_local diffusion: \n', self.gpu_grid_b)

		for i in range(N_STREAMS):

			start_index = i * (STREAM_SIZE // MATRIX_SIZE)
			end_index = start_index + (STREAM_SIZE // MATRIX_SIZE)

			drv.memcpy_htod(self.gpu_grid_a, self.grid[start_index:end_index])

			self.events[i]['non_local_kernel_begin'].record(self.streams[i])
			self.non_local_diffusion(
				self.gpu_grid_b, self.gpu_grid_a, self.randoms, 
				self.random_x_coordinates, self.random_y_coordinates,
				grid = (self.grid_dims, self.grid_dims, 1),
				block = (self.block_dims, self.block_dims, 1),
				stream = self.streams[i])
			self.events[i]['non_local_kernel_end'].record(self.streams[i])
			self.gpu_grid_a, self.gpu_grid_b = self.gpu_grid_b, self.gpu_grid_a

			drv.memcpy_dtoh(self.grid[start_index:end_index], self.gpu_grid_a)

		print('\nGrid after non_local diffusion: \n', self.gpu_grid_a)

	# unnecessary for FOREST implementation
	def run(self):
		i = 0
		self.reference.record()
		while i < N_ITERS:
			#self.generate_randoms()
			self.local()
			#self.generate_random_coords()
			#self.non_local()
			i += 1
		print('\n')
		for k in range(N_STREAMS):
			print('Local_kernel_begin: ', self.reference.time_till(self.events[k]['local_kernel_begin']))
			print('Local_kernel_end: ', self.reference.time_till(self.events[k]['local_kernel_end']))
			#print('Non_local_kernel_begin: ', self.reference.time_till(self.events[k]['non_local_kernel_begin']))
			#print('Non_local_kernel_end: ', self.reference.time_till(self.events[k]['non_local_kernel_end']))

if __name__ == '__main__':
	Diffusion(MATRIX_SIZE, BLOCK_DIMS, GRID_DIMS, P_LOCAL, P_NON_LOCAL)

