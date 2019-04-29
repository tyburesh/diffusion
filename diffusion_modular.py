
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
P_NON_LOCAL = 0.25 # probability of non-local diffusion
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
		self.initialize_grid()
		self.initialize_gpu_memory()
		self.initialize_streams()
		self.initialize_kernel()
		self.run()	# unnecessary for FOREST implementation

	# Create MATRIX_SIZE x MATRIX_SIZE numpy array and initialize seed
	def initialize_grid(self):
		self.grid = np.zeros((self.size, self.size)).astype(np.float32)
		self.grid[self.size // 2][self.size // 2] = 1 # seed is in the center of the matrix
		self.grid = self.grid.reshape(self.size * self.size)
		self.test = drv.pagelocked_empty_like(self.grid)

	# Allocate memory on GPU
	def initialize_gpu_memory(self):
		self.gpu_grid_a = drv.mem_alloc(10000000) # input window
		self.gpu_grid_b = drv.mem_alloc(10000000) # output window
		self.buf_a = drv.mem_alloc(10000000) # input window back buffer
		self.buf_b = drv.mem_alloc(10000000) # output window back buffer
		#self.buf_a_ready = 0
		#self.buf_b_ready = 0

	# Create kernel and kernel functions
	def initialize_kernel(self):
		self.kernel_code = """

			__global__ void local_diffuse(float* grid_a, float* grid_b, float* randoms)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;			// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;			// row element of index

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

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;			// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;			// row element of index

				if (x < grid_size && y < grid_size) {{

					unsigned int thread_id = y * grid_size + x; 					// thread index in array

					if (grid_a[thread_id] == 1) {{
						grid_b[thread_id] = 1;									// current cell
						if (randoms[thread_id] < prob) {{
							unsigned int spread_index = y_coords[thread_id] * grid_size + x_coords[thread_id];
							grid_b[spread_index] = 1;
						}}
					}}
				}}
			}}
		"""
		self.kernel = self.kernel_code.format(self.size, self.p_local, self.size, self.p_non_local, self.size)
		self.mod = SourceModule(self.kernel)
		self.local_diffusion = self.mod.get_function('local_diffuse')
		self.non_local_diffusion = self.mod.get_function('non_local_diffuse')

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
		self.randoms = curandom.rand((self.size, self.size))

	# Generate random coordinates between [0, MATRIX_SIZE - 1]
	# Dictate non_local diffusion location (if it occurs)
	def generate_random_coords(self):
		self.random_x_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)
		self.random_y_coordinates = ((curandom.rand((self.size, self.size))) * self.size).astype(np.int32)

	# Performs one iteration of local diffusion
	def local(self):
		print('\nGrid before local diffusion: {}\n\n'.format(self.grid))

		for i in range(N_STREAMS):

			start_index = i * STREAM_SIZE
			end_index = start_index + STREAM_SIZE
			#start_index = i * (STREAM_SIZE // MATRIX_SIZE)
			#end_index = start_index + (STREAM_SIZE // MATRIX_SIZE)

			# Copy data from host to device
			drv.memcpy_htod_async(self.gpu_grid_a, self.test[start_index:end_index], stream = self.streams[i])
			print('Grid data for iteration {}: {}\n'.format(i, self.test[start_index:end_index]))

			# look into pass start_index and end_index into the kernel function?
			self.events[i]['local_kernel_begin'].record(self.streams[i])
			self.local_diffusion(
				self.gpu_grid_a, self.gpu_grid_b, self.randoms,
				grid = (self.grid_dims, self.grid_dims, 1),
				block = (self.block_dims, self.block_dims, 1),
				stream = self.streams[i],)
			self.events[i]['local_kernel_end'].record(self.streams[i])
			self.gpu_grid_a, self.gpu_grid_b = self.gpu_grid_b, self.gpu_grid_a

			# Copy data from device to host
			drv.memcpy_dtoh_async(self.test[start_index:end_index], self.gpu_grid_a, stream = self.streams[i])
			print('Grid after iteration {}: {}\n\n'.format(i, self.grid))


		#self.grid = self.grid.reshape((self.size, self.size))
		print('\nGrid after local diffusion: \n', self.test)

	# Performs one iteration of non_local diffusion
	def non_local(self):
		print('\nGrid before non_local diffusion: {}\n'.format(self.grid))

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
			self.generate_randoms()
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

