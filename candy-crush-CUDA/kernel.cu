#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <queue>
#include <cstdlib>
#include <iostream>
#include <curand_kernel.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

//CONTROL DE ERRORES
int colores = 6;
int vidas = 5;
__constant__ int dev_pos[2];


__device__ bool esta_conectado(int posInicial, int posObjetivo, int* matriz, int dev_M, int dev_N)
{
    if (posInicial == posObjetivo) return true;
    if (matriz[posInicial] != matriz[posObjetivo]) return false;

    int mov_adyacentes[4] = { 1, -1, dev_N, -dev_N }; // Array de posibles movimientos que se pueden hacer para buscar casillas contiguas
    int* visitados = new int[dev_M * dev_N]; // Array para guardar las posiciones ya visitadas
    for (int i = 0; i < dev_M * dev_N; i++) {
        visitados[i] = 0;
    }
    int* visitados_temp = new int[dev_M * dev_N];  // Array temporal para marcar visitados en la siguiente iteración
    for (int i = 0; i < dev_M * dev_N; i++) {
        visitados_temp[i] = 0;
    }
    int* posiciones_por_visitar = new int[dev_M * dev_N]; // Array para guardar las posiciones a visitar en la siguiente iteración 
    int num_posiciones_por_visitar = 0; // Número de posiciones a visitar en la siguiente iteración

    visitados[posInicial] = 1;
    visitados_temp[posInicial] = 1;
    posiciones_por_visitar[num_posiciones_por_visitar++] = posInicial;

    while (num_posiciones_por_visitar > 0) {
        int pos_actual = posiciones_por_visitar[--num_posiciones_por_visitar];

        for (int i = 0; i < 4; i++) {
            int adyacente = pos_actual + mov_adyacentes[i];

            if (adyacente >= 0 && adyacente < dev_M * dev_N) { //si no nos hemos salido de la matriz       
                if (adyacente / dev_N == pos_actual / dev_N || adyacente % dev_N == pos_actual % dev_N) { //si no hemos saltado de linea con un desplazamiento horizontal (se comprueba si esta en la misma fila o la misma columna)
                    if (visitados[adyacente] != 1) { // si aun no hemos pasado por esa posición
                        if (matriz[posObjetivo] == matriz[adyacente]) {
                            if (adyacente == posObjetivo) {
                                return true;
                            }
                            else {
                                visitados[adyacente] = 1;
                                visitados_temp[adyacente] = 1;
                                posiciones_por_visitar[num_posiciones_por_visitar++] = adyacente;
                            }
                        }
                    }
                }
            }
        }

        // Marcar como visitados las posiciones de la iteración actual
        for (int i = 0; i < dev_M * dev_N; i++) {
            visitados[i] = visitados[i] | visitados_temp[i];
            visitados_temp[i] = 0;
        }
    }

    return false;
}


__device__ int rellenar(int posInicial, int* matriz, unsigned long long seed, int colores, int dev_M, int dev_N)
{
    curandState state;
    curand_init(seed, threadIdx.x, 0, &state);

    int primerCero = (posInicial % dev_N) + (dev_M * (dev_N - 1)); //Nos colocamos en la posición más baja de la columna de la posicion que queremos rellenar
    while (matriz[primerCero] != 0 && primerCero >= 0) {
        primerCero = primerCero - dev_N;
    }   //con esto tenemos la posición del primer cero que se encuentra en la columna
    if (primerCero < 0 || primerCero < posInicial) { //si no hay ceros o estan por encima de la posicion que se quiere rellenar se deja como estaba
        return matriz[posInicial];
    }
    int pos_desde_primer_cero = ((primerCero - posInicial) / dev_N) + 1;  //posiciones en vertical desde el primer cero hasta la posicion que se quiere rellenar
    //printf("%d ", pos_desde_primer_cero);
    int resultado = 0;
    int posActual = primerCero;
    for (int i = 0; i < pos_desde_primer_cero; i++) {   //buscamos el numero en la posicion pos_desde_primer_cero sin contar los ceros, que será la que le corresponda a la posición que queremos al bajar todos los bloques
        while (posActual >= 0 && matriz[posActual] == 0) {    //saltamos los ceros o si nos hemos salido de la matriz
            posActual = posActual - dev_N;
        }
        if (posActual < 0) {
            int aleatorio = curand(&state) % colores + 1;
            return aleatorio;
        }
        resultado = matriz[posActual];
        posActual = posActual - dev_N;
    }
    return resultado;
}



__global__ void addKernel(int* dev_tablero, int* dev_resultado, int* dev_bloques_eliminados, int fila_o_columna, unsigned long long seed, int colores, int dev_M, int dev_N)
{
    //Calculamos la posición en la que va a operar el hilo
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //Si se encuentra dentro de los limites de la matriz hace los calculos
    if (i < dev_M && j < dev_N) {
        int pos_hilo = i * dev_N + j;
        int pos_seleccionada = dev_pos[0] * dev_N + dev_pos[1];
        bool borrar = false;
        int valor_propio_inicial = dev_tablero[pos_hilo]; //ayuda a reducir los accesos a memoria global
        extern __shared__ int shared_tablero[];
        shared_tablero[threadIdx.x * blockDim.y + threadIdx.y] = valor_propio_inicial;
        __syncthreads();

        //Buscar que bloques se van a eliminar
        int valor_pos_seleccionada;
        int i_seleccionada = dev_pos[0];
        int j_seleccionada = dev_pos[1];
        /*
        * queremos saber si el bloque seleccionado se encuentra en el mismo bloque de hilos que el que ejecuta esto para saber si aceder a memoria compartida o global
        * i_seleccionada >= blockIdx.x * blockDim.x; comprueba que esta por debajo de la fila superior del bloque
        * i_seleccionada < blockIdx.x * blockDim.x + blockDim.x; comprueba que esta por encima de la fila inferior del bloque
        * 
        * j_seleccionada >= blockIdx.y * blockDim.y; comprueba que esta a la derecha de la columna con menor indice
        * j_seleccionada < blockIdx.y * blockDim.y + blockDim.y; comprueba que esta a la izquierda de la columna con mayor indice
        */
        if (i_seleccionada >= blockIdx.x * blockDim.x && i_seleccionada < blockIdx.x * blockDim.x + blockDim.x
            && j_seleccionada >= blockIdx.y * blockDim.y && j_seleccionada < blockIdx.y * blockDim.y + blockDim.y)
        {
            valor_pos_seleccionada = shared_tablero[(i_seleccionada%blockDim.x)*blockDim.y+(j_seleccionada%blockDim.y)];
        }
        else {
            valor_pos_seleccionada = dev_tablero[pos_seleccionada];
        }

        if (valor_pos_seleccionada == 8) { // seleccionado bloque bomba
            if (fila_o_columna == 0) // 0->fila, 1->columna
            {
                if (pos_hilo / dev_N == pos_seleccionada / dev_N) borrar = true;
            }
            else
            {
                if (pos_hilo % dev_N == pos_seleccionada % dev_N) borrar = true;
            }
        }
        else if (valor_pos_seleccionada == 9) { //seleccionado bloque TNT
            if (abs(pos_seleccionada / dev_N - pos_hilo / dev_N) <= 4 && abs(pos_seleccionada % dev_N - pos_hilo % dev_N) <= 4) {
                borrar = true;
            }
        }
        else if (valor_pos_seleccionada > 10) { //seleccionado bloque rompecabezas
            if (valor_propio_inicial == valor_pos_seleccionada % 10)
            {
                borrar = true;
            }
        }
        else { //seleccionado bloque normal
            borrar = esta_conectado(pos_hilo, pos_seleccionada, dev_tablero, dev_M, dev_N);
        }
        __syncthreads(); //fin buscar bloques que se van a eliminar
        if (borrar && pos_seleccionada != pos_hilo) { //el bloque seleccionado siempre tendrá borrar = true pero solo hay que borrarlo si hay algún otro bloque conectado así que se hará despues
            dev_resultado[pos_hilo] = 0;
            atomicAdd(&dev_bloques_eliminados[0], 1);
        }
        else {
            dev_resultado[pos_hilo] = valor_propio_inicial;
        }
        __syncthreads(); //fin eliminar bloques normales
        //Colocar bloques especiales si hace falta
        if (pos_seleccionada == pos_hilo && valor_pos_seleccionada > 6) { // si se seleccionó un bloque especial no se va colocar otro aunque se hayan borrado 5 bloques o más
            atomicAdd(&dev_bloques_eliminados[0], 1);
            dev_resultado[pos_hilo] = 0;
        }
        else if (pos_seleccionada == pos_hilo && dev_bloques_eliminados[0] > 0) { // si se seleccionó un bloque normal colocamos bloques especiales en función del número de bloques eliminados en el turno
            atomicAdd(&dev_bloques_eliminados[0], 1);
            if (dev_bloques_eliminados[0] < 5) {
                dev_resultado[pos_hilo] = 0;
            }
            else if (dev_bloques_eliminados[0] == 5) {
                dev_resultado[pos_hilo] = 8; //bomba
            }
            else if (dev_bloques_eliminados[0] == 6) {
                dev_resultado[pos_hilo] = 9; //tnt
            }
            else if (dev_bloques_eliminados[0] >= 7) {
                curandState state;
                curand_init(seed, threadIdx.x, 0, &state);
                int color_rompecabezas = curand(&state) % colores + 1;
                dev_resultado[pos_hilo] = 10 + color_rompecabezas; //rompecabezas
            }
        }
        __syncthreads(); //fin colocar bloques especiales

        int valor_rellenar = rellenar(pos_hilo, dev_resultado, seed, colores, dev_M, dev_N);
        __syncthreads();
        dev_resultado[pos_hilo] = valor_rellenar;
        __syncthreads(); //fin rellenar

    }
}

void print_help() {
    printf("Usage: programa.exe [OPCIONES] DIFICULTAD FILAS COLUMNAS\n");
    printf("Dificultad = 1 -> 4 colores\nDificultad = 2 -> 6 colores\n");
    printf("Opciones:\n");
    printf("  -a,    Use automatic mode\n");
    printf("  -m,    Use difficult mode (default)\n");
}

int main(int argc, char* argv[])
{
    cudaFree(0);
    srand(time(0));

    //gestión de los parámetros de ejecución
    bool automatico = false;
    int dificultad = 2;
    int filas = 10, columnas = 10;
    for (int i = 1; i < argc; i++) { //ignoramos el primero (es el nombre del programa no nos interesa) y lo hacemos una vez por cada argumento
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "-m") == 0) { //automático o manual
            if (strcmp(argv[i], "-a") == 0) automatico = true;
        }
        else if (i == argc - 3 && (atoi(argv[i]) == 1 || atoi(argv[i]) == 2)) { // dificultad 1 o 2
            dificultad = atoi(argv[i]);
            if (dificultad == 1)
            {
                colores = 4;
            }
            else {
                colores = 6;
            }
        }
        else if (i == argc - 2) { //numero de filas
            filas = atoi(argv[i]);
        }
        else if (i == argc - 1) { // numero de columnas
            columnas = atoi(argv[i]);
        }
        else {
            printf("Invalid argument: %s\n", argv[i]);
            print_help();
            return 1;
        }
    }


    //Iniciamos el tablero
    int h_M = filas;
    int h_N = columnas;
    int* h_tablero = new int[h_M * h_N];
    for (int i = 0; i < h_M * h_N; i++) {
        h_tablero[i] = (rand() % colores) + 1;

    }

    //Lo mostramos

    printf("\n -- Tablero Inicial --\n");
    for (int x = 0; x < h_M * h_N; x++) {
        if (h_tablero[x] == 8)
        {
            printf("B  ");
        }
        else if (h_tablero[x] == 9) {
            printf("T  ");
        }
        else if (h_tablero[x] >= 10)
        {
            printf("R%d ", (h_tablero[x]) % 10);
        }
        else {
            printf("%d  ", h_tablero[x]);
        }
        if (x % h_N == h_N - 1) printf("\n");
    }

    while (vidas > 0)
    {
        //Pedimos al usuario que indique su movimiento
        int x_seleccionada;
        int y_seleccionada;
        bool es_entero;

        if (automatico)
        {
            printf("Haciendo jugada automatica\n");
            x_seleccionada = (rand() % h_M);
            y_seleccionada = (rand() % h_N);
            printf("Casilla: { %d, %d}\n", x_seleccionada, y_seleccionada);
        }
        else
        {
            do { //coordenada x
                cout << "Introduce x: ";
                cin >> x_seleccionada;

                es_entero = !cin.fail(); // No es un número entero.

                if (!es_entero || x_seleccionada >= h_M) {
                    cin.clear(); // Limpia el error de cin.
                    cin.ignore(10000, '\n'); // Ignora todos los caracteres no válidos que se ingresaron.
                    cout << "Error: debes ingresar un numero entero entre 0 y " << h_M - 1 << endl;
                }
            } while (!es_entero || x_seleccionada >= h_M);

            do { //coordenada y
                cout << "Introduce y: ";
                cin >> y_seleccionada;

                es_entero = !cin.fail(); // No es un número entero.

                if (!es_entero || y_seleccionada >= h_N) {
                    cin.clear(); // Limpia el error de cin.
                    cin.ignore(10000, '\n'); // Ignora todos los caracteres no válidos que se ingresaron.
                    cout << "Error: debes ingresar un numero entero entre 0 y " << h_N - 1 << endl;
                }
            } while (!es_entero || x_seleccionada >= h_N);
        }

        int h_pos[2] = { x_seleccionada, y_seleccionada };


        //Reservamos memoria para las matrices y las copiamos al Device
        int* h_resultado = new int[h_M * h_N];
        //int h_resultado[M][N];
        int* dev_tablero = new int[h_M * h_N];
        int* dev_resultado = new int[h_M * h_N];
        //int(*dev_tablero)[N], (*dev_resultado)[N];
        cudaMalloc((void**)&dev_tablero, h_N * h_M * sizeof(int));
        cudaMalloc((void**)&dev_resultado, h_N * h_M * sizeof(int));

        cudaMemcpy(dev_tablero, h_tablero, h_M * h_N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_resultado, h_resultado, h_M * h_N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(dev_pos, h_pos, 2 * sizeof(int));


        int h_bloques_eliminados[1] = { 0 };
        int(*dev_bloques_eliminados)[1];
        cudaMalloc((void**)&dev_bloques_eliminados, sizeof(int));
        cudaMemcpy(dev_bloques_eliminados, h_bloques_eliminados, sizeof(int), cudaMemcpyHostToDevice);

        int color_rompecabezas = (rand() % colores) + 1;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int MAX_HILOS = prop.maxThreadsPerBlock;
        // Mientras no se supere MAX_HILOS todo va bien
        int bloques_x = 2;
        int bloques_y = 2;
        int hilos_x = h_M / bloques_x;
        if (hilos_x * bloques_x < h_M) hilos_x++;
        int hilos_y = h_N / bloques_y;
        if (hilos_y * bloques_y < h_M) hilos_y++;
        int tam_shared = hilos_x* hilos_y * sizeof(int);

        //Iniciamos hilos y bloques
        dim3 blocksInGrid(bloques_x, bloques_y);
        dim3 threadsInBlock(hilos_x, hilos_y);
        addKernel << <blocksInGrid, threadsInBlock, tam_shared >> > (dev_tablero, dev_resultado, *dev_bloques_eliminados, (rand() % 2), (unsigned long long) time(NULL), colores, h_M, h_N);
        //rand() % 2 es un numero aleatorio entre 0 y 1 para saber si se borra la fila o la columna al encontrar una bomba, se pasa desde el host porque debe ser el mismo para todos los hilos

        cudaMemcpy(h_bloques_eliminados, dev_bloques_eliminados, sizeof(int), cudaMemcpyDeviceToHost);
        //Copiamos el resultado al host
        cudaMemcpy(h_resultado, dev_resultado, h_M * h_N * sizeof(int), cudaMemcpyDeviceToHost);

        //Y lo mostramos
        for (int x = 0; x < h_M * h_N; x++) {
            if (h_resultado[x] == 8)
            {
                printf("B  ");
            }
            else if (h_resultado[x] == 9)
            {
                printf("T  ");
            }
            else if (h_resultado[x] >= 10)
            {
                printf("R%d ", (h_resultado[x]) % 10);
            }
            else
            {
                printf("%d  ", h_resultado[x]);
            }
            if (x % h_N == h_N - 1) printf("\n");
        }

        printf("Bloques eliminados en este movimiento: %d\n", h_bloques_eliminados[0]);
        if (h_bloques_eliminados[0] == 0) {
            vidas--;
            printf("Has perdido una vida, te quedan %d\n", vidas);
        }

        for (int i = 0; i < h_M * h_N; i++) h_tablero[i] = h_resultado[i];

        //Por ultimo liberamos memoria
        cudaFree(dev_tablero); cudaFree(dev_resultado); cudaFree(dev_pos); cudaFree(dev_bloques_eliminados);
    }

    printf("Has perdido");

    return 0;
}