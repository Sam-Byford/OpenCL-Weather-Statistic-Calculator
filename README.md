# OpenCL-Weather-Statistic-Calculator

To run the code simply click the local windows debugger inside visual studio (2019 or later) to launch and run the application. Two varients of the algorithm will then be executed, one that uses optimised kernels to calculate statistics and one that does not.
The performance of both algorithms will then be displayed at the end of the program and a comparison of their times will be visible.

This program calculates the mean, minimum, maximum, and standard deviation of a supplied lincolnshire weather dataset. It begins by extracting the temperature values as floats from the dataset. The program is then split into two sections; 
The first section contains the ‘optimised’ methods and calculates the statistical values in the most optimised way possible. The second section contains the ‘Non-optimised’ methods and calculates the statistics in a very in-efficient manner.

The main optimisations used were to utilise local storage through creating local copies of the input vectors and splitting the vectors into workgroups. The workgroup size was 32 as this was stated as the preferred size when the kernels were queried. 
By splitting the workload into 32 groups the parallelism of the application was increased and therefore, so was its speed. Other optimisations included automatically reducing the size of output arrays when a kernel was called recursively so as little memory as possible was used. 
Atomic functions were used sparingly but, in some cases, they were proven to be more efficient than recursion. When recursion was used, it was only used up to the point where the output was less than 1000 - The final calculations were done sequentially. 
This saved resources as the transferring of so few items to and from a kernel would have taken longer than running it sequentially.


























