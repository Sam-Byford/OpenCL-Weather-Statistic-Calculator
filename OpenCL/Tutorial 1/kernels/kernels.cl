//***Mean***
kernel void reduce_atomic_single_array(global const float* Temperatures, global int* Output, local float* localCopy) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	localCopy[lid] = Temperatures[id]; //create local copy for quicker accessing 

	barrier(CLK_GLOBAL_MEM_FENCE); //ensure all threads copy

	for (int stride=1; stride<N; stride*=2) {
		if ((lid % (stride*2) == 0)) {
			localCopy[lid] += localCopy[lid+stride];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	//copy the cache to output array
	if (!lid) {
		float intpart;
		float fractpart = modf (localCopy[lid] , &intpart); // use modf function to extract the decimal value
		int convertedInt = convert_int(localCopy[lid]); // convert the float value to int (remove everything after decimal point)
		int convertedDesc = convert_int(fractpart*10); // multiply by 10 to turn the decimal point to a float > 0. Then convert to int. only multipled by 10 as only
		//require one point of precision

		// add the two values to two seperate positions - one for the integer values and one for the decimal values
		atomic_add(&Output[0],convertedInt);
		atomic_add(&Output[1],convertedDesc);
	}
}

//***Min***
kernel void min_reduce(global const float* Temperatures, global float* Output_min, local float* localCopy){
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	localCopy[lid] = Temperatures[id]; //create local copy for quicker accessing 

	barrier(CLK_GLOBAL_MEM_FENCE); //ensure all threads copy

	for (int stride=1; stride<N; stride*=2) {
		if ((lid % (stride*2) == 0) && ((lid + stride) < N)) {
			if (localCopy[lid] > localCopy[lid+stride]){
				localCopy[lid] = localCopy[lid+stride];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	//copy the cache to output array
	if (!lid) {
		int idOffsetMultiplier = id/32;
		Output_min[idOffsetMultiplier] = localCopy[lid];
	}

}

//***Max***
kernel void max_reduce(global const float* Temperatures, global float* Output_max, local float* localCopy){
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	localCopy[lid] = Temperatures[id]; //create local copy for quicker accessing 

	barrier(CLK_GLOBAL_MEM_FENCE); //ensure all threads copy

	for (int stride=1; stride<N; stride*=2) {
		if ((lid % (stride*2) == 0) && ((lid + stride) < N)) {
			if (localCopy[lid] < localCopy[lid+stride]){
				localCopy[lid] = localCopy[lid+stride];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	//copy the cache to output array
	if (!lid) {
		int idOffsetMultiplier = id/32;
		Output_max[idOffsetMultiplier] = localCopy[lid];
	}
}

//***SD***
kernel void sd_map(global const float* Temperatures, global float* Output_sd, global float* Mean, local float* localCopy){
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	localCopy[lid] = Temperatures[id]; //create local copy for quicker accessing 
	float mean = Mean[0];

	barrier(CLK_GLOBAL_MEM_FENCE); //ensure all threads copy

	float meanSub = localCopy[lid] - mean;
	float val = pow(meanSub, 2);
	Output_sd[id] = val;
}

kernel void reduce(global const float* Temperatures, global float* Output_reduce, local float* localCopy) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	localCopy[lid] = Temperatures[id]; //create local copy for quicker accessing 

	barrier(CLK_GLOBAL_MEM_FENCE); //ensure all threads copy

	for (int stride=1; stride<N; stride*=2) {
		if ((lid % (stride*2) == 0)) {
			localCopy[lid] += localCopy[lid+stride];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	//copy the cache to output array
	if (!lid) {
		//in order to store the results of each workgroup reduction in the output array and not overwrite it with each group...
		//...an offsetmultiplier is needed. This value works out which group has just completed by dividing the global id by 32.
		//It then places the result of the workgroup at that position in the output array.
		//E.g. group 1 will have an offset = 32/32 = 1 and therefore be placed at index 1, group 2 will have offset = 64/32 = 2 and will be placed at index 2...
		int idOffsetMultiplier = id/32;
		Output_reduce[idOffsetMultiplier] = localCopy[lid];
	}
}

kernel void reduce_2(global const float* Temperatures, global int* Output_reduce, local float* localCopy) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	localCopy[lid] = Temperatures[id]; //create local copy for quicker accessing 

	barrier(CLK_GLOBAL_MEM_FENCE); //ensure all threads copy

	for (int stride=1; stride<N; stride*=2) {
		if ((lid % (stride*2) == 0)) {
			localCopy[lid] += localCopy[lid+stride];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (!lid) {
		float intpart;
		float fractpart = modf (localCopy[lid] , &intpart);
		int convertedInt = convert_int(localCopy[lid]);
		int convertedDesc = convert_int(fractpart*10); 

		// add the two values to two seperate arrays - one for the integer values and one for the decimal values
		atomic_add(&Output_reduce[0],convertedInt);
		atomic_add(&Output_reduce[1],convertedDesc);
	}
}

//***Median***
//Bitonic sort kernels - Only work with small vectors
void swap_bitonic(global float* A, global float* B, bool dir) {
	if ((!dir && *A > *B) || (dir && *A < *B)) {
		//printf("Sort Order: %d\n", dir);
		float t = *A;
		*A = *B;
		*B = t;
	}
}

void bitonic_merge(int id, global float* A, int N, bool dir) {
	for (int i = N/2; i > 0; i/=2) {
		if ((id % (i*2)) < i) {
			swap_bitonic(&A[id],&A[id+i],dir);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

kernel void sort_bitonic(global float* A, global float *Output) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = 1; i <= N/2; i*=2) {
		if (id % (i*4) < i*2) {
			bitonic_merge(id, A, i*2, false);
		}
		else if (((id + i*2) % (i*4)) < i*2) {
			bitonic_merge(id, A, i*2, true);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	bitonic_merge(id,A,N,false);
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[id] = A[id];
}

