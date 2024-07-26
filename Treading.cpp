#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CONSTANTS
#define PI 3.14159265358979323846 // Pi value
#define NUM_SP 1000               // Number of Spatial Points
#define NUM_TS 10000              // Number of Time Steps
#define THERMAL_DIFF 0.01         // Thermal Diffusivity
#define SPATIAL_SS 0.01           // Spatial Step Size
#define TIME_SS 0.0001            // Time Step Size

//Method for setting up the Temperature Distribution (Done with Sine waves)
void setUp(double* u, int numSP) {
    for (int i = 0; i < numSP; i++) {
        u[i] = sin(PI * i * SPATIAL_SS);
    }
}

//Method for solving the Heat Equation (Finite Version)
void solverHeat(double* u, int numSP, int numTS, double thermalDiff, double spatialSS, double timeSS) {
    double* u_new = (double*)malloc(numSP * sizeof(double)); //Allocate memory for the array

    for (int t = 0; t < numTS; t++) { //Time Step Loop
#pragma omp parallel for
        for (int i = 1; i < numSP - 1; i++) { //Loop for updating the temperature in each Spatial Point
            u_new[i] = u[i] + thermalDiff * timeSS / (spatialSS * spatialSS) * (u[i + 1] - 2 * u[i] + u[i - 1]);
        }
#pragma omp parallel for
        for (int i = 1; i < numSP - 1; i++) { //Sets the new Temperature to the original array
            u[i] = u_new[i];
        }
    }
    free(u_new);
}

//Method for printing the temperature values for readability
void printValues(double* u, int numSP, const char* title) {
    printf("%s:\n", title);
    printf("Index\tTemperature\n");
    printf("-----\t-----------\n");
    for (int i = 0; i < numSP; i++) {
        printf("%d\t%f\n", i, u[i]);
    }
    printf("\n");
}

int main() {
    double* u = (double*)malloc(NUM_SP * sizeof(double));
    setUp(u, NUM_SP); //Initialize the temperature distribution

    //INITIAL TEMP DISTRIBUTION
    printf("Initial Temperature Distribution:\n");
    printValues(u, NUM_SP, "Initial");

    //Measure execution time for the solver
    double start_time = omp_get_wtime();
    solverHeat(u, NUM_SP, NUM_TS, THERMAL_DIFF, SPATIAL_SS, TIME_SS);
    double end_time = omp_get_wtime();

    //FINAL TEMP DISTRIBUTION
    printf("Final Temperature Distribution:\n");
    printValues(u, NUM_SP, "Final");

    //EXECUTION TIME
    printf("---------------------------------------------------\n");
    printf("Execution time: %f seconds\n", end_time - start_time);

    free(u);
    return 0;
}

//INFO GATHERED:
// For omp multi threading, when the number of threads for each loop is 4 and the executioni time is 0.031962 second. And in Performance Profiler is 3.512 second
// Without omp multi threading, The execute time in Performance Profiler is 4.493 second

/*
When having a larger spread of Spartial Points it is better to
use treads so that there can be many loops done, about 4-5 is the correct amount
in smaller case 5 is best, and in bigger cases 4 is best.
Overall, not using thread is the worst option available.
*/