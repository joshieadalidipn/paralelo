#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846
#define N_TERMS 11  // Número de términos de la serie de Fourier
#define N 1000  // Número de puntos en la discretización de la integral

__device__ double f(double x) {
    return pow(x, 4) - 5 * pow(x, 2) - 2 * x + 1;
}

__global__ void calculate_coefficients_kernel(double *a, double *b, int n_terms, double L, double h) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n > 0 && n <= n_terms) {
        double integral_a = 0.0;
        double integral_b = 0.0;
        double x;

        for (int i = 0; i <= N; i++) {
            x = -L + i * h;
            integral_a += f(x) * cos(n * PI * x / L) * (i == 0 || i == N ? 0.5 : 1.0);
            integral_b += f(x) * sin(n * PI * x / L) * (i == 0 || i == N ? 0.5 : 1.0);
        }

        a[n] = integral_a * h / L;
        b[n] = integral_b * h / L;
    }
}

void calculate_coefficients(double *a, double *b, int n_terms, double L) {
    double h = (2 * L) / N;

    double *d_a, *d_b;
    cudaMalloc((void**)&d_a, (n_terms + 1) * sizeof(double));
    cudaMalloc((void**)&d_b, n_terms * sizeof(double));
    cudaMemset(d_a, 0, (n_terms + 1) * sizeof(double));
    cudaMemset(d_b, 0, n_terms * sizeof(double));

    // Calcular a_0 en el host
    a[0] = 0;
    for (int i = 0; i <= N; i++) {
        double x = -L + i * h;
        a[0] += f(x) * (i == 0 || i == N ? 0.5 : 1);
    }
    a[0] *= h / (2 * L);

    int blockSize = 256;
    int numBlocks = (n_terms + blockSize) / blockSize;
    calculate_coefficients_kernel<<<numBlocks, blockSize>>>(d_a, d_b, n_terms, L, h);
    cudaDeviceSynchronize();

    cudaMemcpy(a + 1, d_a + 1, n_terms * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, n_terms * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
}

double fourier_approximation(double x, const double *a, const double *b, int n_terms, double L) {
    double sum = a[0] / 2;  // a_0/2 es el término constante
    for (int n = 1; n <= n_terms; n++) {
        sum += a[n] * cos(n * PI * x / L) + b[n] * sin(n * PI * x / L);
    }
    return sum;
}

void export_to_csv(const char *filename, double *a, double *b, double L) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    // Escribir el encabezado del CSV
    fprintf(file, "x,y,y_fourier\n");

    double x, y, y_fourier, step = 4 * PI / N;
    for (int i = 0; i <= N; i++) {
        x = -2 * PI + i * step;
        y = f(x);
        y_fourier = fourier_approximation(x, a, b, N_TERMS, L);
        fprintf(file, "%f,%f,%f\n", x, y, y_fourier);
    }
    fclose(file);
}

int main() {
    double a[N_TERMS + 1] = {0};
    double b[N_TERMS] = {0};

    calculate_coefficients(a, b, N_TERMS, 2 * PI);
    export_to_csv("fourier_data.csv", a, b, 2 * PI);

    return 0;
}
