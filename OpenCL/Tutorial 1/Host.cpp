//#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <list>
#include <numeric>
#include <iostream>
#include <algorithm> 
#include <math.h>  
#include <chrono>  // for high_resolution_clock

#include "Utils.h"

using namespace std;

//General Methods
vector<float> padding(vector<float> Temperatures, size_t workgroupSize, int initValue, bool sorting) {
    
    if (!sorting) {
        size_t padding_size = Temperatures.size() % workgroupSize;

        //if the input vector is not a multiple of the workgroupSize
        //insert additional neutral elements (0 for addition) so that the total will not be affected
        if (padding_size) {
            std::vector<float> Temp_ext(workgroupSize - padding_size, initValue);
            //append extra vector to our inputs
            Temperatures.insert(Temperatures.end(), Temp_ext.begin(), Temp_ext.end());
        }
    }
    //This section is triggered if we are using this padding method to pad a to-be-sorted array
    //if we are we, not only do we need to make sure the vector length is a factor of 32 but also a power of 2 as...
    //...bitonic sort only works with lengths of a power of 2. '32768' was used for the short dataset as its the closest
    //...power of 2 and multiple of 32 to the vectors original length 
    else {
        size_t padding_size = 32768 - Temperatures.size();
        std::vector<float> Temp_ext(padding_size, initValue);
        //append extra vector to our inputs
        Temperatures.insert(Temperatures.end(), Temp_ext.begin(), Temp_ext.end());
    }

    return Temperatures;
}

void readFile(vector<float>& Temperatures_unpadded) {

    // Read from the text file
    ifstream file("temp_lincolnshire_datasets/temp_lincolnshire.txt");
    string line;

    cout << "******READING FILE*******" << endl;
    cout << "Extracting temperatures from file, will take around 30 seconds..." << endl;
    // Use a while loop together with the getline() function to read the file line by line
    while (getline(file, line)) {
        //split the contents of the line to extract the temperature value
        size_t found = line.find_last_of(' ');
        string tempStr = line.substr(found);
        float temp = atof(tempStr.c_str());
        Temperatures_unpadded.push_back(temp);
    }

    file.close();

    cout << "Temperatures extracted sucessfully" << endl;
}



//Optimised Methods
void minimum(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize, cl::CommandQueue queue,
    int &Kernel_time, int &Total_mem_time, int &Overall_time, int counter) {
    // Find the min element
    // pad the vector so its length is adequate for the number of work groups
    // value '1000' is used so the new padded elements dont get in the way of finding the min
    vector<float> Temperatures_min = padding(Temperatures_unpadded, workgroupSize, 1000, false);

    size_t vector_elements = Temperatures_min.size();
    size_t vector_size = Temperatures_min.size() * sizeof(float);

    //with each reduction the output produced is the input divided by 32, therefore for efficieny we re-size the output vector...
    //...on each run. Means there is no wasted memory.
    std::vector<float> Output_min(vector_elements/32, 1000); 
    size_t output_size_min = Output_min.size() * sizeof(float);

    cl::Buffer buffer_Temp_min(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out_min(context, CL_MEM_READ_WRITE, output_size_min);

    cl::Event write_event;

    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp_min, CL_TRUE, 0, vector_size, &Temperatures_min[0], NULL, &write_event);
    queue.enqueueFillBuffer(buffer_Out_min, 0, 0, output_size_min);

    // setup kenerl
    cl::Kernel kernel_min = cl::Kernel(program, "min_reduce");
    kernel_min.setArg(0, buffer_Temp_min);
    kernel_min.setArg(1, buffer_Out_min);
    kernel_min.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(workgroupSize), NULL, &kernel_event);

    cl::Event read_event;

    // Read output of kernel
    queue.enqueueReadBuffer(buffer_Out_min, CL_TRUE, 0, output_size_min, &Output_min[0], NULL, &read_event);

    //calculate performance times
    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int Current_Kernel_Time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();;
    int Current_mem_time = write_time + read_time;

    Kernel_time += kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    Total_mem_time += write_time + read_time;
    Overall_time += Current_mem_time + Current_Kernel_Time;

    // reduce temperatures vector an additional 3 times until there are < 64 items left in vector
    if (counter < 2) {
        counter++;
        minimum(Output_min, context, program, workgroupSize, queue, Kernel_time, Total_mem_time, Overall_time, counter);
    }
    // when there are only a handful of items left in vector it is not efficient to run min calculation in parallel. The time taken to transfer data to device and execute kernel >
    // ...the time taken to calculate the min sequentially. Therefore we simply calculate the min from this small sample size sequentially
    else {
        float minTemp = 1000;
        for (int k = 0; k < Output_min.size(); ++k) {
            if (Output_min[k] < minTemp) {
                minTemp = Output_min[k];
            }
        }
        cout << "Calculated Min = " << minTemp << endl;
    }
}

void mean(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize,
    cl::CommandQueue queue, float& Mean, int& Kernel_time, int& Total_mem_time, int& Overall_time, bool optimised) {
    //mean value is passed by reference so it can be altered and used later on in the SD calculations
    cout << "\n******MEAN******" << endl;
    cout << "Note: This function is identical on the optimised and non-optimised algorithm varients" << endl;
    // The mean is calculated in parallel by splitting the reduction results into integer and decimal components
    // The kernel approach used was one output array where Output[0] would contain integer reduction and Output[1] the decimal reduction
    // Unfortunatley I could not find any futhur optimisations to this approach and therefore both the optimised and non-optimised varients utilise this same function
    vector<float> Temperatures = padding(Temperatures_unpadded, workgroupSize, 0, false); 
    //array padded with 0's to make sure its the right size for the workgroup size - 0s wont affect the mean calculation

    size_t vector_elements = Temperatures.size();//number of elements
    size_t vector_size = Temperatures.size() * sizeof(float);

    std::vector<int> Output(2); // only has two elements - Int sum and decimal sum
    size_t output_size = Output.size() * sizeof(float);//size in bytes

    // Buffers
    cl::Buffer buffer_Temp(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out(context, CL_MEM_READ_WRITE, output_size);

    cl::Event write_event;

    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp, CL_TRUE, 0, vector_size, &Temperatures[0], NULL, &write_event);
    queue.enqueueFillBuffer(buffer_Out, 0, 0, output_size); //zero buffer on device memory

    // setup kenerl
    cl::Kernel kernel_reduce = cl::Kernel(program, "reduce_atomic_single_array");
    kernel_reduce.setArg(0, buffer_Temp);
    kernel_reduce.setArg(1, buffer_Out);
    kernel_reduce.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(workgroupSize), NULL, &kernel_event);

    cl::Event read_event;

    // Read output of kernel
    queue.enqueueReadBuffer(buffer_Out, CL_TRUE, 0, output_size, &Output[0], NULL, &read_event);

    //sequentially calculate mean
    //...no need for this to be parallel as its quick n simple. Parallel would actually slow it down due to copying of data
    float decimal_sum = ((float)Output[1]) / 10;
    Mean = ((float)Output[0] + decimal_sum) / vector_elements;

    cout << "\nCalculated Mean: ";
    printf("%.1f", Mean);

    Kernel_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
        kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    Total_mem_time = write_time + read_time;
    Overall_time = Total_mem_time + Kernel_time;

    std::cout << "\n\nKernel execution time [ns]: " << Kernel_time << std::endl;
    std::cout << "Total memory transfer time [ns]: " << Total_mem_time << std::endl;
    std::cout << "Overall Opetation Time [ns]: " << Overall_time << std::endl;

    std::cout << GetFullProfilingInfo(kernel_event, ProfilingResolution::PROF_US) << std::endl;
}

void maximum(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize, cl::CommandQueue queue,
    int& Kernel_time, int& Total_mem_time, int& Overall_time, int counter) {
    //pad vector with -1000 so it is an adequate size for the number of work groups. values of -1000 will not get in the way of calculating max
    vector<float> Temperatures_max = padding(Temperatures_unpadded, workgroupSize, -1000, false);

    size_t vector_elements = Temperatures_max.size();
    size_t vector_size = Temperatures_max.size() * sizeof(float);

    //with each reduction the output produced is the input divided by 32, therefore for efficieny we re-size the output vector...
    //...on each run. Means there is no wasted memory.
    std::vector<float> Output_max(vector_elements / 32, 1000);
    size_t output_size_max = Output_max.size() * sizeof(float);

    cl::Buffer buffer_Temp_max(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out_max(context, CL_MEM_READ_WRITE, output_size_max);

    cl::Event write_event;

    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp_max, CL_TRUE, 0, vector_size, &Temperatures_max[0], NULL, &write_event);
    queue.enqueueFillBuffer(buffer_Out_max, 0, 0, output_size_max);

    // setup kenerl
    cl::Kernel kernel_max = cl::Kernel(program, "max_reduce");
    kernel_max.setArg(0, buffer_Temp_max);
    kernel_max.setArg(1, buffer_Out_max);
    kernel_max.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(workgroupSize), NULL, &kernel_event);

    cl::Event read_event;

    // Read output of kernel
    queue.enqueueReadBuffer(buffer_Out_max, CL_TRUE, 0, output_size_max, &Output_max[0], NULL, &read_event);

    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int Current_Kernel_Time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();;
    int Current_mem_time = write_time + read_time;

    Kernel_time += kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    Total_mem_time += write_time + read_time;
    Overall_time += Current_mem_time + Current_Kernel_Time;

    if (counter < 2) {
        counter++;
        maximum(Output_max, context, program, workgroupSize, queue, Kernel_time, Total_mem_time, Overall_time, counter);
    }
    else {
        float maxTemp = -1000;
        for (int k = 0; k < Output_max.size(); ++k) {
            if (Output_max[k] > maxTemp) {
                maxTemp = Output_max[k];
            }
        }
        cout << "Calculated Max = " << maxTemp << endl;
    }
}

void reduce_add_non_optimised(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize, cl::CommandQueue queue,
    int& Kernel_time, int& Total_mem_time, int& Overall_time, int counter, float sampleSize) {
    //Step 2
    vector<float> Temperatures_reduce = padding(Temperatures_unpadded, workgroupSize, 0, false);

    size_t vector_elements = Temperatures_reduce.size();
    size_t vector_size = Temperatures_reduce.size() * sizeof(float);

    std::vector<float> Output_reduce(vector_elements, 1000);
    size_t output_size_reduce = Output_reduce.size() * sizeof(float);

    cl::Buffer buffer_Temp_reduce(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out_reduce(context, CL_MEM_READ_WRITE, output_size_reduce);

    cl::Event write_event;

    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp_reduce, CL_TRUE, 0, vector_size, &Temperatures_reduce[0], NULL, &write_event);
    queue.enqueueFillBuffer(buffer_Out_reduce, 0, 0, output_size_reduce);

    // setup kenerl
    cl::Kernel kernel_sd = cl::Kernel(program, "reduce");
    kernel_sd.setArg(0, buffer_Temp_reduce);
    kernel_sd.setArg(1, buffer_Out_reduce);
    kernel_sd.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_sd, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(workgroupSize), NULL, &kernel_event);

    cl::Event read_event;

    // Read output of kernel
    queue.enqueueReadBuffer(buffer_Out_reduce, CL_TRUE, 0, output_size_reduce, &Output_reduce[0], NULL, &read_event);

    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int Current_Kernel_Time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();;
    int Current_mem_time = write_time + read_time;

    Kernel_time += kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    Total_mem_time += write_time + read_time;
    Overall_time += Current_mem_time + Current_Kernel_Time;

    //run the reduction pattern again to futhur reduce the vector
    if (counter < 5) {
        counter++;
        reduce_add_non_optimised(Output_reduce, context, program, workgroupSize, queue, Kernel_time, Total_mem_time, Overall_time, counter, sampleSize);
    }
    else {
        //Step 3
        //with the sum complete, divide by sample size and square root for final SD
        float sd = sqrt((Output_reduce[0] / sampleSize));
        cout << "Calculated SD = ";
        printf("%.1f",sd);
    }
}

void reduce_add_optimised(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize, cl::CommandQueue queue,
    int& Kernel_time, int& Total_mem_time, int& Overall_time, int counter, float sampleSize) {
    //Step 2

    //This kernel is more efficient than its counterpart due to it not having excess transfer and memory time caused by running...
    //...multiple kernels. Instead it uses atomic functions which, although in-efficient, actually prove to be better for this dataset
    vector<float> Temperatures_reduce = padding(Temperatures_unpadded, workgroupSize, 0, false);

    size_t vector_elements = Temperatures_reduce.size();
    size_t vector_size = Temperatures_reduce.size() * sizeof(float);

    std::vector<int> Output_reduce(2);
    size_t output_size_reduce = Output_reduce.size() * sizeof(float);

    cl::Buffer buffer_Temp_reduce(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out_reduce(context, CL_MEM_READ_WRITE, output_size_reduce);

    cl::Event write_event;

    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp_reduce, CL_TRUE, 0, vector_size, &Temperatures_reduce[0], NULL, &write_event);
    queue.enqueueFillBuffer(buffer_Out_reduce, 0, 0, output_size_reduce);

    // setup kenerl
    cl::Kernel kernel_sd = cl::Kernel(program, "reduce_2");
    kernel_sd.setArg(0, buffer_Temp_reduce);
    kernel_sd.setArg(1, buffer_Out_reduce);
    kernel_sd.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_sd, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(workgroupSize), NULL, &kernel_event);

    cl::Event read_event;
#
    // Read output of kernel
    queue.enqueueReadBuffer(buffer_Out_reduce, CL_TRUE, 0, output_size_reduce, &Output_reduce[0], NULL, &read_event);

    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int Current_Kernel_Time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();;
    int Current_mem_time = write_time + read_time;

    Kernel_time += kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    Total_mem_time += write_time + read_time;
    Overall_time += Current_mem_time + Current_Kernel_Time;

    //Step 3
    float decimal_sum = ((float)Output_reduce[1]) / 10;
    float sumSq = ((float)Output_reduce[0] + decimal_sum);
    //with the sum complete, divide by sample size and square root for final SD
    float sd = sqrt((sumSq / sampleSize));
    cout << "Calculated SD = ";
    printf("%.1f", sd);
}

void sd(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize, cl::CommandQueue queue,
    int& Kernel_time, int& Total_mem_time, int& Overall_time, float Mean, float sampleSize, bool optimised) {
    // The SD is a three step process
    // 1. Use the map pattern to calculate (each item - mean)^2
    // 2. Take these new values and perform a reduction pattern to add them together
    // 3. sequentially calculate the SD by dividing this sum by the number of items in the un-padded vector and square rooting the answer

    // Step 1
    // save the unpadded size so we can divide the sum later on by the correct value
    size_t vector_elements_before_padding = Temperatures_unpadded.size();
    vector<float> Temperatures_sd = padding(Temperatures_unpadded, workgroupSize, 0, false);

    size_t vector_elements = Temperatures_sd.size();
    size_t vector_size = Temperatures_sd.size() * sizeof(float);

    std::vector<float> Output_sd(vector_elements, 0);
    size_t output_size_sd = Output_sd.size() * sizeof(float);

    std::vector<float> Mean_sd = { Mean };
    size_t mean_size_sd = Mean_sd.size() * sizeof(float);

    cl::Buffer buffer_Temp_sd(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out_sd(context, CL_MEM_READ_WRITE, output_size_sd);
    cl::Buffer buffer_Mean_sd(context, CL_MEM_READ_WRITE, mean_size_sd);

    cl::Event write_event;
    cl::Event write_event_2;

    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp_sd, CL_TRUE, 0, vector_size, &Temperatures_sd[0], NULL, &write_event);
    queue.enqueueWriteBuffer(buffer_Mean_sd, CL_TRUE, 0, mean_size_sd, &Mean_sd[0], NULL, &write_event_2);
    queue.enqueueFillBuffer(buffer_Out_sd, 0, 0, output_size_sd);

    // setup kenerl
    cl::Kernel kernel_sd = cl::Kernel(program, "sd_map");
    kernel_sd.setArg(0, buffer_Temp_sd);
    kernel_sd.setArg(1, buffer_Out_sd);
    kernel_sd.setArg(2, buffer_Mean_sd);
    kernel_sd.setArg(3, cl::Local(workgroupSize * sizeof(float)));//local memory size

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_sd, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(workgroupSize), NULL, &kernel_event);

    cl::Event read_event;

    // Read output of kernel
    queue.enqueueReadBuffer(buffer_Out_sd, CL_TRUE, 0, output_size_sd, &Output_sd[0], NULL, &read_event);

    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int write_time_2 = write_event_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event_2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int Current_Kernel_Time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();;
    int Current_mem_time = write_time + write_time_2 + read_time;

    Kernel_time += kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    Total_mem_time += write_time + read_time + write_time_2;
    Overall_time += Current_mem_time + Current_Kernel_Time;

    //remove padded values so they not included in the reduce addition sum - prevents skewed result
    int sizeDiff = vector_elements - vector_elements_before_padding;
    Output_sd.resize(vector_elements - sizeDiff);

    //each workgroups sum of (item - mean)^2 has been calculated. Combine these sums with reduce pattern
    if (optimised) {
        reduce_add_optimised(Output_sd, context, program, workgroupSize, queue, Kernel_time, Total_mem_time, Overall_time, 0, sampleSize);
    }
    else {
        reduce_add_non_optimised(Output_sd, context, program, workgroupSize, queue, Kernel_time, Total_mem_time, Overall_time, 0, sampleSize);
    }
}

void bitonic(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize, cl::CommandQueue queue) {
    //This method attempts to sort the array using 'bitonic sort'. The sorted aray would have been used for median etc
    //Unfortunatley I could only get the sort working for smaller arrays (length <= 512) and therefore I could not
    //run this method successfully.
    
    // pad the array to have length equal to a multiple of 32 AND a power of 2 (bitonic only works with a power of 2)
    vector<float> Temperatures = padding(Temperatures_unpadded, 0, 1000, true); 
    size_t vector_elements = Temperatures.size();//number of elements
    size_t vector_size = Temperatures.size() * sizeof(float);

    std::vector<float> Output_bitonic(vector_elements, 0);
    size_t output_size_bitonic = Output_bitonic.size() * sizeof(float);

    // Buffers
    cl::Buffer buffer_Temp(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out_bitonic(context, CL_MEM_READ_WRITE, output_size_bitonic);

    cl::Event write_event;

    cout << "Input Size: " << Temperatures.size() << endl; //double check input size
    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp, CL_TRUE, 0, vector_size, &Temperatures[0], NULL, &write_event);
    queue.enqueueFillBuffer(buffer_Out_bitonic, 0, 0, output_size_bitonic);

    // setup kenerl
    cl::Kernel kernel_bitonic = cl::Kernel(program, "sort_bitonic");
    kernel_bitonic.setArg(0, buffer_Temp);
    kernel_bitonic.setArg(1, buffer_Out_bitonic);
    //kernel_bitonic.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_bitonic, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &kernel_event);

    cl::Event read_event;

    queue.enqueueReadBuffer(buffer_Out_bitonic, CL_TRUE, 0, output_size_bitonic, &Output_bitonic[0], NULL, &read_event);

    //cout << "\nUn-Sorted list: " << Temperatures << endl;
    std::cout << "Sorted list: " << Output_bitonic << endl;

    //Calculate performance of kernel 
    int Kernel_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int Total_mem_time = write_time + read_time;

    std::cout << "\n\nKernel execution time [ns]: " << Kernel_time << std::endl;
    std::cout << "Total memory transfer time [ns]: " << Total_mem_time << std::endl;
    std::cout << "Overall Opetation Time [ns]: " << Total_mem_time + Kernel_time << std::endl;

    std::cout << GetFullProfilingInfo(kernel_event, ProfilingResolution::PROF_US) << std::endl;
}

void execute_optimised_program(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program,
    size_t workgroupSize, cl::CommandQueue queue, int& Total_Kernel_time, int& Total_mem_time, int& Total_program_time) {
    //**********MEAN**********  
    float meanVal = 0;
    int Kernel_time_mean = 0;
    int Total_mem_time_mean = 0;
    int Overall_time_mean = 0;
    mean(Temperatures_unpadded, context, program, workgroupSize, queue, meanVal, Kernel_time_mean, Total_mem_time_mean, Overall_time_mean, true);

    //***********MINIMUM**********      
    cout << "\n******MINIMUM******" << endl;

    //call the min_reduce function multiple times to perform multi-pass reduction     
    int Kernel_time_min = 0;
    int Total_mem_time_min = 0;
    int Overall_time_min = 0;
    minimum(Temperatures_unpadded, context, program, workgroupSize, queue, Kernel_time_min, Total_mem_time_min, Overall_time_min, 0);

    std::cout << "\nKernel execution time [ns]: " << Kernel_time_min << std::endl;
    std::cout << "Total memory transfer time [ns]: " << Total_mem_time_min << std::endl;
    std::cout << "Overall Opetation Time [ns]: " << Overall_time_min << std::endl;

    //**********MAXIMUM**********
    cout << "\n******MAXIMUM******" << endl;

    int Kernel_time_max = 0;
    int Total_mem_time_max = 0;
    int Overall_time_max = 0;
    //cout << "Actual Max = " << *std::max_element(std::begin(Temperatures_unpadded), std::end(Temperatures_unpadded)) << endl;
    maximum(Temperatures_unpadded, context, program, workgroupSize, queue, Kernel_time_max, Total_mem_time_max, Overall_time_max, 0);

    std::cout << "\nKernel execution time [ns]: " << Kernel_time_max << std::endl;
    std::cout << "Total memory transfer time [ns]: " << Total_mem_time_max << std::endl;
    std::cout << "Overall Opetation Time [ns]: " << Overall_time_max << std::endl;

    //**********Standard Deviation
    cout << "\n******STANDARD DEVIATION******" << endl;

    // Gunna run this two ways, with atomic and without (recursive)
    int Kernel_time_sd = 0;
    int Total_mem_time_sd = 0;
    int Overall_time_sd = 0;
    int sampleSize = Temperatures_unpadded.size();

    sd(Temperatures_unpadded, context, program, workgroupSize, queue, Kernel_time_sd, Total_mem_time_sd, Overall_time_sd, meanVal, sampleSize, true);

    //int Kernel_time_sd_atomic = 0;
    //int Total_mem_time_sd_atomic = 0;
    //int Overall_time_sd_atomic = 0;        
    //sd_atomic(Temperatures_unpadded, context, program, workgroupSize, queue);

    std::cout << "\nKernel execution time [ns]: " << Kernel_time_sd << std::endl;
    std::cout << "Total memory transfer time [ns]: " << Total_mem_time_sd << std::endl;
    std::cout << "Overall Opetation Time [ns]: " << Overall_time_sd << std::endl;

    //std::cout << "\nKernel execution time atomic [ns]: " << Kernel_time_sd_atomic << std::endl;
    //std::cout << "Total memory transfer time atomic [ns]: " << Total_mem_time_sd_atomic << std::endl;
    //std::cout << "Overall Opetation Time atomic [ns]: " << Overall_time_sd_atomic << std::endl;

    //**Bitonic test**
    //cout << "\n******BITONIC SORT******" << endl;
    //bitonic(Temperatures_unpadded, context, program, workgroupSize, queue, 0);

    //**Total Performance Metrics**
    Total_Kernel_time = Kernel_time_min + Kernel_time_max + Kernel_time_sd + Kernel_time_mean;
    Total_mem_time = Total_mem_time_min + Total_mem_time_max + Total_mem_time_sd + Total_mem_time_mean;
    Total_program_time = Overall_time_min + Overall_time_max + Overall_time_sd + Overall_time_mean;
}



//Non-optimised methods
void minimum_non_optimised(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize, cl::CommandQueue queue,
    int& Kernel_time, int& Total_mem_time, int& Overall_time, int counter) {
    // Finds the minimum element the non-optimised way. Code is almost identical to the optimised version bar a few tweaks
    vector<float> Temperatures_min = padding(Temperatures_unpadded, workgroupSize, 1000, false);

    size_t vector_elements = Temperatures_min.size();
    size_t vector_size = Temperatures_min.size() * sizeof(float);

    std::vector<float> Output_min(vector_elements, 1000); // non-optimised version keeps the output vector the same length
    size_t output_size_min = Output_min.size() * sizeof(float);

    cl::Buffer buffer_Temp_min(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out_min(context, CL_MEM_READ_WRITE, output_size_min);

    cl::Event write_event;

    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp_min, CL_TRUE, 0, vector_size, &Temperatures_min[0], NULL, &write_event);
    queue.enqueueFillBuffer(buffer_Out_min, 0, 0, output_size_min);

    // setup kenerl
    cl::Kernel kernel_min = cl::Kernel(program, "min_reduce");
    kernel_min.setArg(0, buffer_Temp_min);
    kernel_min.setArg(1, buffer_Out_min);
    kernel_min.setArg(2, cl::Local(workgroupSize * sizeof(float)));

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(workgroupSize), NULL, &kernel_event);

    cl::Event read_event;

    // Read output of kernel
    queue.enqueueReadBuffer(buffer_Out_min, CL_TRUE, 0, output_size_min, &Output_min[0], NULL, &read_event);

    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int Current_Kernel_Time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();;
    int Current_mem_time = write_time + read_time;

    Kernel_time += kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    Total_mem_time += write_time + read_time;
    Overall_time += Current_mem_time + Current_Kernel_Time;

    // non-optimised version runs the reduction more, after these 5 runs the output array will contain the min...
    if (counter < 5) {
        counter++;
        minimum(Output_min, context, program, workgroupSize, queue, Kernel_time, Total_mem_time, Overall_time, counter);
    }
    //...no need for running sequentially over the vector as it has been reduced enough times 
    else {
        cout << "Calculated Min = " << Output_min[0] << endl;
    }
}

void maximum_non_optimised(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, size_t workgroupSize, cl::CommandQueue queue,
    int& Kernel_time, int& Total_mem_time, int& Overall_time, int counter) {
    //The following is almost identical to the minimum_non_optimised method - Major differences are the array is padded...
    //...with '-1000' instead of '1000' and 'max_reduce' is the kernel called, not 'min_reduce'
    vector<float> Temperatures_max = padding(Temperatures_unpadded, workgroupSize, -1000, false);

    size_t vector_elements = Temperatures_max.size();
    size_t vector_size = Temperatures_max.size() * sizeof(float);

    std::vector<float> Output_max(vector_elements, 1000);
    size_t output_size_max = Output_max.size() * sizeof(float);

    cl::Buffer buffer_Temp_max(context, CL_MEM_READ_WRITE, vector_size);
    cl::Buffer buffer_Out_max(context, CL_MEM_READ_WRITE, output_size_max);

    cl::Event write_event;

    // copy to device memory
    queue.enqueueWriteBuffer(buffer_Temp_max, CL_TRUE, 0, vector_size, &Temperatures_max[0], NULL, &write_event);
    queue.enqueueFillBuffer(buffer_Out_max, 0, 0, output_size_max);

    // setup kenerl
    cl::Kernel kernel_max = cl::Kernel(program, "max_reduce");
    kernel_max.setArg(0, buffer_Temp_max);
    kernel_max.setArg(1, buffer_Out_max);
    kernel_max.setArg(2, cl::Local(workgroupSize * sizeof(float)));//local memory size

    cl::Event kernel_event;

    // execute kernel
    queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(workgroupSize), NULL, &kernel_event);

    cl::Event read_event;

    // Read output of kernel
    queue.enqueueReadBuffer(buffer_Out_max, CL_TRUE, 0, output_size_max, &Output_max[0], NULL, &read_event);

    int write_time = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    int read_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    int Current_Kernel_Time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();;
    int Current_mem_time = write_time + read_time;

    Kernel_time += kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    Total_mem_time += write_time + read_time;
    Overall_time += Current_mem_time + Current_Kernel_Time;

    if (counter < 5) {
        counter++;
        maximum(Output_max, context, program, workgroupSize, queue, Kernel_time, Total_mem_time, Overall_time, counter);
    }
    else {
        cout << "Calculated Max = " << Output_max[0] << endl;
    }
}

void execute_non_optimised_program(std::vector<float> Temperatures_unpadded, cl::Context context, cl::Program program, 
    size_t workgroupSize, cl::CommandQueue queue, int& Total_Kernel_time, int& Total_mem_time, int& Total_program_time){
    //code alsmost identical to 'execute_non_optimised_program' except we are now running the methods for the non-optimised algorithm

    //**********MEAN**********  
    float meanVal = 0;
    int Kernel_time_mean = 0;
    int Total_mem_time_mean = 0;
    int Overall_time_mean = 0;
    mean(Temperatures_unpadded, context, program, workgroupSize, queue, meanVal, Kernel_time_mean, Total_mem_time_mean, Overall_time_mean, false);

    //***********MINIMUM**********      
    cout << "\n******MINIMUM******" << endl;

    //call the min_reduce function multiple times to perform multi-pass reduction     
    int Kernel_time_min = 0;
    int Total_mem_time_min = 0;
    int Overall_time_min = 0;
    minimum_non_optimised(Temperatures_unpadded, context, program, workgroupSize, queue, Kernel_time_min, Total_mem_time_min, Overall_time_min, 0);

    std::cout << "\nKernel execution time [ns]: " << Kernel_time_min << std::endl;
    std::cout << "Total memory transfer time [ns]: " << Total_mem_time_min << std::endl;
    std::cout << "Overall Opetation Time [ns]: " << Overall_time_min << std::endl;

    //**********MAXIMUM**********
    cout << "\n******MAXIMUM******" << endl;

    int Kernel_time_max = 0;
    int Total_mem_time_max = 0;
    int Overall_time_max = 0;
    //cout << "Actual Max = " << *std::max_element(std::begin(Temperatures_unpadded), std::end(Temperatures_unpadded)) << endl;
    maximum_non_optimised(Temperatures_unpadded, context, program, workgroupSize, queue, Kernel_time_max, Total_mem_time_max, Overall_time_max, 0);

    std::cout << "\nKernel execution time [ns]: " << Kernel_time_max << std::endl;
    std::cout << "Total memory transfer time [ns]: " << Total_mem_time_max << std::endl;
    std::cout << "Overall Opetation Time [ns]: " << Overall_time_max << std::endl;

    //**********Standard Deviation
    cout << "\n******STANDARD DEVIATION******" << endl;

    float var = 0;
    for (int n = 0; n < Temperatures_unpadded.size(); n++)
    {
        var += (Temperatures_unpadded[n] - meanVal) * (Temperatures_unpadded[n] - meanVal);
    }
    var /= Temperatures_unpadded.size();
    float sd_act = sqrt(var);
    cout << "Actual SD: ";
    printf("%.1f\n", sd_act);

    // Gunna run this two ways, with atomic and without (recursive)
    int Kernel_time_sd = 0;
    int Total_mem_time_sd = 0;
    int Overall_time_sd = 0;
    int sampleSize = Temperatures_unpadded.size();

    sd(Temperatures_unpadded, context, program, workgroupSize, queue,Kernel_time_sd, Total_mem_time_sd, Overall_time_sd, meanVal, sampleSize, false);

    //int Kernel_time_sd_atomic = 0;
    //int Total_mem_time_sd_atomic = 0;
    //int Overall_time_sd_atomic = 0;        

    std::cout << "\nKernel execution time [ns]: " << Kernel_time_sd << std::endl;
    std::cout << "Total memory transfer time [ns]: " << Total_mem_time_sd << std::endl;
    std::cout << "Overall Opetation Time [ns]: " << Overall_time_sd << std::endl;

    //**Bitonic sort**
    //cout << "\n******BITONIC SORT******" << endl;
    //bitonic(Temperatures_unpadded, context, program, workgroupSize, queue, 0);

    //**Total Performance Metrics**
    Total_Kernel_time = Kernel_time_min + Kernel_time_max + Kernel_time_sd + Kernel_time_mean;
    Total_mem_time = Total_mem_time_min + Total_mem_time_max + Total_mem_time_sd + Total_mem_time_mean;
    Total_program_time = Overall_time_min + Overall_time_max + Overall_time_sd + Overall_time_mean;


}

//Program Entry
/*
This programme calculates the mean, minimum, maximum, and standard deviation of the supplied dataset. 
It begins by extracting the temperature values as floats from the dataset. The programme is then split into two sections. 
The first section contains the ‘optimised’ methods and calculates the statistical values in the most optimised way possible. 
The second section contains the ‘Non-optimised’ methods and calculates the statistics in a very in-efficient manner.
Original developments include the way I have gotten around atomic_add() not allowing floats. 
I split the reduced value into integer and decimal values. 
The decimal value was multiplied by 10 to convert to an integer (only 10 as we only care about 1 d.p.). 
Then both values were atomically added to the vector separately. 
When complete the final reduction equalled the decimal value (converted to a float and divided by 10) plus the integer value (converted to a float).

The main optimisations used were to utilise local storage through creating local copies of the input vectors and splitting the vectors into workgroups. 
The workgroup size was 32 as this was stated as the preferred size when the kernels were queried. 
By splitting the workload into 32 groups the parallelism of the application was increased and therefore, so was its speed. 
Other optimisations included automatically reducing the size of output arrays when a kernel was called recursively so as little memory as possible was used. 
Atomic functions were used sparingly but, in some cases, they were proven to be more efficient than recursion.
When recursion was used, it was only used up to the point where the output was less than 1000 - The final calculations were done sequentially. 
This saved resources as the transferring of so few items to and from a kernel would have taken longer than running it sequentially.
*/
int main()
{    
    try {
        //hardcoded to use the devices GPU due to its parallel abilities
        int platform_id = 1;
        int device_id = 0;

        cl::Context context = GetContext(platform_id, device_id);
        std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

        cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
        cl::Program::Sources sources;
        AddSources(sources, "kernels/kernels.cl");
        cl::Program program(context, sources);

        //build and debug the kernel code
        try {
            program.build();
        }
        catch (const cl::Error& err) {
            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        vector<float> Temperatures_unpadded;

        readFile(Temperatures_unpadded);

        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // get device
        size_t workgroupSize = 32;//Value found by running - kernel_reduce.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);...
        //...in basic implementation. Cant actually use this line in code as kernel not defined yet

        //**********OPTIMISED PROGRAM**********
        cout << "\n--------------------------------------Executing Optimised Program--------------------------------------" << endl;
        int Total_Kernel_time_O = 0; //_O = optimised
        int Total_mem_time_O = 0;
        int Total_program_time_O = 0;
        execute_optimised_program(Temperatures_unpadded, context, program, workgroupSize, queue, Total_Kernel_time_O, Total_mem_time_O, Total_program_time_O);

        //bitonic(Temperatures_unpadded,context,program,workgroupSize,queue);

        //**********NON-OPTIMISED PROGRAM*********
        cout << "\n--------------------------------------Executing Non-Optimised Program--------------------------------------" << endl;
        int Total_Kernel_time_NO = 0; //_NO = non-optimised
        int Total_mem_time_NO = 0;
        int Total_program_time_NO = 0;
        execute_non_optimised_program(Temperatures_unpadded, context, program, workgroupSize, queue, Total_Kernel_time_NO, Total_mem_time_NO, Total_program_time_NO);

        //Program Performance output
        cout << "\n--------------------------------------Program Performance Comparison--------------------------------------" << endl;

        cout << "\n*******TOTAL OPTIMISED PROGRAM PERFORMANCE METRICS*******" << endl;
        std::cout << "\nOverall Kernel execution time [ns]: " << Total_Kernel_time_O << std::endl;
        std::cout << "Overall memory transfer time [ns]: " << Total_mem_time_O << std::endl;
        std::cout << "Total Program Opetation Time [ns]: " << Total_program_time_O << std::endl;

        cout << "\n*******TOTAL NON-OPTIMISED PROGRAM PERFORMANCE METRICS*******" << endl;
        std::cout << "\nOverall Kernel execution time [ns]: " << Total_Kernel_time_NO << std::endl;
        std::cout << "Overall memory transfer time [ns]: " << Total_mem_time_NO << std::endl;
        std::cout << "Total Program Opetation Time [ns]: " << Total_program_time_NO << std::endl;
      
        int timeSaved = Total_program_time_NO - Total_program_time_O;
        cout << "\nToal time saved with optimisations [ns]: " << timeSaved << endl;
    }
    catch (cl::Error err) {
        cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
    }

    return 0;
}