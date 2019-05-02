
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
N_STREAMS = 2 # data broken into four quadrants
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

	# Allocate host memory
	# Host memory must be pagelocked because kernel execution and data transfers are overlapped 
	def initialize_grid(self):
		self.grid = drv.pagelocked_empty((self.size, self.size), np.float32)
		self.grid[1][1] = 1 # seed is in the center of the matrix
		self.grid[6][6] = 1
		self.grid = self.grid.reshape(self.size * self.size) # 2D grid becomes 1D grd

	# Allocate device memory
	# Target window size is roughly 25% of GPU memory
	def initialize_gpu_memory(self):
		nbytes = (self.grid.nbytes + 1) // 2
		self.gpu_grid_a = drv.mem_alloc(nbytes) # input window
		self.gpu_grid_b = drv.mem_alloc(nbytes) # output window
		#self.buf_a = drv.mem_alloc(10000000) # input window back buffer
		#self.buf_b = drv.mem_alloc(10000000) # output window back buffer
		#self.buf_a_ready = 0
		#self.buf_b_ready = 0

	# Create kernel functions
	# These functions will be run in parallel on GPU
	def initialize_kernel(self):
		self.kernel_code = """

			__global__ void local_diffuse(float* grid_a, float* grid_b, float* randoms)
			{{

				unsigned int grid_size = {};
				float prob = {};

				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;				// column element of index
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;				// row element of index

				// make sure cell is within the grid dimensions
				if (x < grid_size && y < grid_size) {{

					unsigned int thread_id = y * grid_size + x; 					// thread index in array
					printf("Thread ID = %u\\n", thread_id);

					// edges will be ignored as starting points
					unsigned int edge = (x == 0) || (x == grid_size - 1) || (y == 0) || (y == grid_size - 1);

					// only look at this cell if it is already a 1
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

				// make sure cell is within the grid dimensions
				if (x < grid_size && y < grid_size) {{

					unsigned int thread_id = y * grid_size + x; 					// thread index in array

					// only look at this cell if it is already a 1
					if (grid_a[thread_id] == 1) {{
						grid_b[thread_id] = 1;										// current cell
						if (randoms[thread_id] < prob) {{
							unsigned int spread_index = y_coords[thread_id] * grid_size + x_coords[thread_id];
							grid_b[spread_index] = 1;
						}}
					}}
				}}
			}}
		"""

		# format kernel code with constants and compile kernel code
		self.kernel = self.kernel_code.format(self.size, self.p_local, self.size, self.p_non_local, self.size)
		self.mod = SourceModule(self.kernel)

		# grab diffusion functions from kernel code
		self.local_diffusion = self.mod.get_function('local_diffuse')
		self.non_local_diffusion = self.mod.get_function('non_local_diffuse')

	# Create PyCUDA streams and events
	# Streams are needed because kernel execution and data transfers are overlapped
	def initialize_streams(self):
		self.streams = []
		self.start_event = drv.Event()
		self.end_event = drv.Event()
		for k in range(N_STREAMS):
			self.streams.append(drv.Stream())

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

		self.start_event.record()
		for i in range(N_STREAMS):

			# grid slice that will be passed into kernel this iteration
			start_index = i * STREAM_SIZE
			end_index = start_index + STREAM_SIZE
			#start_index = i * (STREAM_SIZE // MATRIX_SIZE)
			#end_index = start_index + (STREAM_SIZE // MATRIX_SIZE)

			# copy data from host to device
			drv.memcpy_htod_async(int(self.gpu_grid_a), self.grid[start_index:end_index], stream = self.streams[i])
			
			print('Grid_GPU = {}'.format(int(self.gpu_grid_a)))

			# look into pass start_index and end_index into the kernel function?
			# invoke kernel function
			self.local_diffusion(
				self.gpu_grid_a, self.gpu_grid_b, self.randoms,
				grid = (self.grid_dims, self.grid_dims, 1),
				block = (self.block_dims, self.block_dims, 1),
				stream = self.streams[i],)
			self.gpu_grid_a, self.gpu_grid_b = self.gpu_grid_b, self.gpu_grid_a

			# copy data from device to host
			drv.memcpy_dtoh_async(self.grid[start_index:end_index], int(self.gpu_grid_a), stream = self.streams[i])
			self.streams[i].synchronize()

		# record end of iteration and synchronize streams
		self.end_event.record()
		self.end_event.synchronize()
		self.secs = self.start_event.time_till(end_event)
		print("\nTime elapsed for iteration of local diffusion: {}", self.secs)

		self.grid = self.grid.reshape((self.size, self.size))
		print('\nGrid after local diffusion: \n', self.grid)

	# Performs one iteration of non_local diffusion
	def non_local(self):
		print('\nGrid before non_local diffusion: {}\n'.format(self.grid))

		for i in range(N_STREAMS):

			start_index = i * (STREAM_SIZE // MATRIX_SIZE)
			end_index = start_index + (STREAM_SIZE // MATRIX_SIZE)

			drv.memcpy_htod(self.gpu_grid_a, self.grid[start_index:end_index])

			self.non_local_diffusion(
				self.gpu_grid_b, self.gpu_grid_a, self.randoms, 
				self.random_x_coordinates, self.random_y_coordinates,
				grid = (self.grid_dims, self.grid_dims, 1),
				block = (self.block_dims, self.block_dims, 1),
				stream = self.streams[i])
			self.gpu_grid_a, self.gpu_grid_b = self.gpu_grid_b, self.gpu_grid_a

			drv.memcpy_dtoh(self.grid[start_index:end_index], self.gpu_grid_a)

		print('\nGrid after non_local diffusion: \n', self.gpu_grid_a)

	# unnecessary for FOREST implementation
	def run(self):
		i = 0
		while i < N_ITERS:
			self.generate_randoms()
			self.local()
			#self.generate_random_coords()
			#self.non_local()
			i += 1
		print('\n')

if __name__ == '__main__':
	Diffusion(MATRIX_SIZE, BLOCK_DIMS, GRID_DIMS, P_LOCAL, P_NON_LOCAL)

