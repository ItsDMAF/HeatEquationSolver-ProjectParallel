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

// Method for setting up the initial Temperature Distribution using sine waves
void setUp(double* u, int numSP) {
    for (int i = 0; i < numSP; i++) {
        u[i] = sin(PI * i * SPATIAL_SS);
    }
}

// Method for solving the Heat Equation using finite difference method
void solverHeat(double* u, int numSP, int numTS, double thermalDiff, double spatialSS, double timeSS) {
    double* u_new = (double*)malloc(numSP * sizeof(double)); // Allocate memory for the new temperature values and make it to double

    for (int t = 0; t < numTS; t++) { // Time Step Loop
#pragma omp parallel for
        for (int i = 1; i < numSP - 1; i++) { // Update temperature
            u_new[i] = u[i] + thermalDiff * timeSS / (spatialSS * spatialSS) * (u[i + 1] - 2 * u[i] + u[i - 1]);
        }
#pragma omp parallel for
        for (int i = 1; i < numSP - 1; i++) { // Copy new temperature values back to the original array
            u[i] = u_new[i];
        }
    }
    free(u_new); // Free the allocated memory
}

// Method for printing the temperature values for readability
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
    double* u = (double*)malloc(NUM_SP * sizeof(double)); // Allocate memory for the temperature array
    setUp(u, NUM_SP); // Initialize the temperature distribution

    // Print initial temperature distribution
    printf("Initial Temperature Distribution:\n");
    printValues(u, NUM_SP, "Initial");

    // Measure and print execution time for the solver
    double start_time = omp_get_wtime();
    solverHeat(u, NUM_SP, NUM_TS, THERMAL_DIFF, SPATIAL_SS, TIME_SS);
    double end_time = omp_get_wtime();

    // Print final temperature distribution
    printf("Final Temperature Distribution:\n");
    printValues(u, NUM_SP, "Final");
    printf("---------------------------------------------------\n");
    printf("Execution time: %f seconds\n", end_time - start_time);

    free(u); // Free the allocated memory function for malloc
    return 0;
}
