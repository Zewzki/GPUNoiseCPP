#include <stdio.h>
#include <cmath>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#define WIDTH 800
#define HEIGHT 600
#define ARR_SIZE 512

double F2 = .5 * (sqrt(3.0) - 1.0);
double G2 = (3.0 - sqrt(3.0)) / 6.0;

__constant__ int iterations = 8;
__constant__ double scale = 0.01;
__constant__ double persistance = 0.6;
__constant__ int high = 255;
__constant__ int low = 0;
__constant__ int w = WIDTH;
__constant__ int h = HEIGHT;

__constant__ int perm[ARR_SIZE];
__constant__ int permMod12[ARR_SIZE];
const int p[] = {151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180 };

__constant__ int grad3[][3] = { { 1, 1, 0 }, { -1, 1, 0 }, { 1, -1, 0 }, { -1, -1, 0 }, { 1, 0, 1 }, { -1, 0, 1 }, { 1, 0, -1 }, { -1, 0, -1 }, { 0, 1, 1 }, { 0, -1, 1 }, { 0, 1, -1 }, { 0, -1, -1 } };

__device__ double mix(double a, double b, double t) {
  return (1 - t) * a + t * b;
}

__device__ double fade(double t) {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ double dot(int g[], double x, double y, double z) {
  return g[0] * x + g[1] * y + g[2] * z;
}

__device__ int fastFloor(double x) {
  return x > 0 ? (int) x : (int) x - 1;
}

__device__ double noise(double x, double y, double z) {

  int X = fastFloor(x);
  int Y = fastFloor(y);
  int Z = fastFloor(z);

  x = x - X;
  y = y - Y;
  z = z - Z;

  X = X & 255;
  Y = Y & 255;
  Z = Z & 255;

  int gi000 = perm[X + perm[Y + perm[Z]]] % 12;
  int gi001 = perm[X + perm[Y + perm[Z + 1]]] % 12;
  int gi010 = perm[X + perm[Y + 1 + perm[Z]]] % 12;
  int gi011 = perm[X + perm[Y + 1 + perm[Z + 1]]] % 12;
  int gi100 = perm[X + 1 + perm[Y + perm[Z]]] % 12;
  int gi101 = perm[X + 1 + perm[Y + perm[Z + 1]]] % 12;
  int gi110 = perm[X + 1 + perm[Y + 1 + perm[Z]]] % 12;
  int gi111 = perm[X + 1 + perm[Y + 1 + perm[Z + 1]]] % 12;

  double n000 = dot(grad3[gi000], x, y, z);
  double n100 = dot(grad3[gi100], x - 1, y, z);
  double n010 = dot(grad3[gi010], x, y - 1, z);
  double n110 = dot(grad3[gi110], x - 1, y - 1, z);
  double n001 = dot(grad3[gi001], x, y, z - 1);
  double n101 = dot(grad3[gi101], x - 1, y, z - 1);
  double n011 = dot(grad3[gi011], x, y - 1, z - 1);
  double n111 = dot(grad3[gi111], x - 1, y - 1, z - 1);

  double u = fade(x);
  double v = fade(y);
  double w = fade(z);

  double nx00 = mix(n000, n100, u);
  double nx01 = mix(n001, n101, u);
  double nx10 = mix(n010, n110, u);
  double nx11 = mix(n011, n111, u);

  double nxy0 = mix(nx00, nx10, v);
  double nxy1 = mix(nx01, nx11, v);
  double nxyz = mix(nxy0, nxy1, w);

  return nxyz;

}

__global__ void sumOctave(int z, sf::Uint8 *result) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  int x = i % blockDim.x;
  int y = (int) (i / blockDim.x);
  
  double maxAmp = 0.0;
  double amp = 1.0;
  double freq = scale;
  double n = 0;
  
  for(int i = 0; i < iterations; i++) {
  
    double adding = noise(x * freq, y * freq, z * freq) * amp;
    n += adding;
    maxAmp += amp;
    amp *= persistance;
    freq *= 2;
  
  }
  
  n /= maxAmp;
  
  n = n * (high - low) / 2 + (high + low) / 2;
  
  result[(i * 4)] = n;
  result[(i * 4) + 1] = n;
  result[(i * 4) + 2] = n;
  //result[(i * 4) + 3] = 255;
  
  //result[i] = n;
  
  //printf("(%d, %d) = %d", x, y, n);

}

int main(void) {

  int N = WIDTH * HEIGHT;
  
  int blockSize = 512;
  int nBlocks = (N / blockSize) + 1;
  
  int * hostPerm;
  int * hostPermMod12;
  
  hostPerm = new int[ARR_SIZE];
  hostPermMod12 = new int[ARR_SIZE];
  
  for(int i = 0; i < ARR_SIZE; i++) {
    hostPerm[i] = p[i & 255];
    hostPermMod12[i] = (hostPerm[i] % 12);
  }
  
  cudaMemcpyToSymbol(perm, hostPerm, ARR_SIZE * sizeof(int));
  cudaMemcpyToSymbol(permMod12, hostPermMod12, ARR_SIZE * sizeof(int));

  // host array
  sf::Uint8 *screen = (sf::Uint8*) malloc(N * 4 * sizeof(sf::Uint8));
  for(int i = 0; i < N * 4; i++) screen[i] = 255;
  
  // device array
  sf::Uint8 *d_screen = new sf::Uint8[WIDTH * HEIGHT * 4];
  cudaMalloc(&d_screen, N * 4 * sizeof(sf::Uint8));
  
  sf::Texture texture;
  if (!texture.create(WIDTH, HEIGHT)) return -1;
  
  sf::Sprite sprite(texture);
  
  int z = 0;
  
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Noise");
  
  cudaMemcpy(d_screen, screen, N * 4 * sizeof(sf::Uint8), cudaMemcpyHostToDevice);
  
  //window.setVerticalSyncEnabled(true);
  
  while(window.isOpen()) {
  
    sf::Event event;
    
    // close window when 'x' is pressed, thus exiting outer loop
    while(window.pollEvent(event)) if (event.type == sf::Event::Closed) window.close();
    
    // call kernel, sync, and copy info off of gpu
    //sumOctave<<<nBlocks, blockSize>>>(z, d_screen);
    sumOctave<<<600, 800>>>(z, d_screen);
    cudaDeviceSynchronize();
    cudaMemcpy(screen, d_screen, N * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);
    
    // update texture
    texture.update(screen);
    window.draw(sprite);
 
    window.display();
    
    z++;
  
  }
  
  cudaFree(d_screen);
  free(screen);
  
  return 0;
  
}
