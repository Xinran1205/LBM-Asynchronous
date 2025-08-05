/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>


/* === Intel‑only intrinsics → GCC 兼容 shim === */
#if defined(__GNUC__)
#  include <stdlib.h>
#  include <stddef.h>
/* __assume(cond) —— GCC 下无等价，直接空吞 */
#  ifndef __assume
#    define __assume(x) ((void)0)
#  endif
/* __assume_aligned(p, a) —— 如果要利用对齐提示可以启用下一行，否则也空吞：
#  ifndef __assume_aligned
#    define __assume_aligned(p, a) p = __builtin_assume_aligned(p, a)
#  endif
*/
#  ifndef __assume_aligned
#    define __assume_aligned(p, a) ((void)0)
#  endif
/* _mm_malloc/_mm_free → posix_memalign/free */
static inline void* _mm_malloc(size_t sz, size_t align) {
    void *ptr = NULL;
    return posix_memalign(&ptr, align, sz) == 0 ? ptr : NULL;
}
#  ifndef _mm_free
#    define _mm_free free
#  endif
#endif
/* === End shim === */




#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"


/* struct to hold the parameter values */
typedef struct
{
    int    nx;            /* no. of cells in x-direction */
    int    ny;            /* no. of cells in y-direction */
    int    maxIters;      /* no. of iterations */
    int    reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */
} t_param;

typedef struct {
    float* speeds0;
    float* speeds1;
    float* speeds2;
    float* speeds3;
    float* speeds4;
    float* speeds5;
    float* speeds6;
    float* speeds7;
    float* speeds8;
} t_speeds_soa;


/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speeds_soa* cells_soa, t_speeds_soa* tmp_cells_soa,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float fusion_more(t_param params, t_speeds_soa* cells_soa, t_speeds_soa* tmp_cells_soa,const int* obstacles, float w11,
                  float w22, float c_sq, float w0, float w1, float w2, float divideVal, float divideVal2);
float av_velocity(t_param params, t_speeds_soa* cells_soa,const int* obstacles);
float calc_reynolds(t_param params, t_speeds_soa* cells_soa, int* obstacles);
float total_density(t_param params, t_speeds_soa* cells_soa);
int write_values(t_param params, t_speeds_soa* cells_soa, int* obstacles, float* av_vels);
int finalise(const t_param* params, t_speeds_soa* cells_soa, t_speeds_soa* tmp_cells_soa, int** obstacles_ptr, float** av_vels_ptr);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    t_param  params;              /* struct to hold parameter values */
    t_speeds_soa cells_soa;    /* grid containing fluid densities */
    t_speeds_soa tmp_cells_soa;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
    struct timeval timstr;                                                             /* structure to hold elapsed time */
    double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

    /* parse the command line */
    if (argc != 3)
    {
        usage(argv[0]);
    }
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    /* Total/init time starts here: initialise our data structures and load values from file */
    gettimeofday(&timstr, NULL);
    tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    init_tic=tot_tic;

    initialise(paramfile, obstaclefile, &params, &cells_soa, &tmp_cells_soa, &obstacles, &av_vels);

    const float w11 = params.density * params.accel / 9.f;
    const float w22 = params.density * params.accel / 36.f;
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */
    const float divideVal = 2.f * c_sq * c_sq;
    const float divideVal2 = 2.f * c_sq;

    /* Init time stops here, compute time starts*/
    gettimeofday(&timstr, NULL);
    init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    comp_tic=init_toc;


    t_speeds_soa *cells_soa_ptr = &cells_soa;
    t_speeds_soa *tmp_cells_soa_ptr = &tmp_cells_soa;

//    __assume(params.maxIters % 2 == 0);
//    __assume(params.maxIters % 4 == 0);
//    __assume(params.maxIters % 10 == 0);
//    __assume(params.maxIters % 20 == 0);
//    __assume(params.maxIters % 40 == 0);
//    __assume(params.maxIters % 100 == 0);
//    __assume(params.maxIters % 200 == 0);
//    __assume(params.maxIters % 400 == 0);
//    __assume(params.maxIters % 1000 == 0);
//    __assume(params.maxIters % 2000 == 0);
//    __assume(params.maxIters % 4000 == 0);
//    __assume(params.maxIters % 10000 == 0);
//    __assume(params.maxIters % 20000 == 0);
//    __assume(params.maxIters % 40000 == 0);
    for(int tt = 0; tt < params.maxIters; tt++)
    {
        av_vels[tt] = fusion_more(params, cells_soa_ptr, tmp_cells_soa_ptr, obstacles,
                                  w11, w22, c_sq, w0, w1, w2,divideVal,divideVal2);

        // swap the pointers
        t_speeds_soa *temp_ptr = cells_soa_ptr;
        cells_soa_ptr = tmp_cells_soa_ptr;
        tmp_cells_soa_ptr = temp_ptr;
    }

    /* Compute time stops here, collate time starts*/
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic=comp_toc;

    // Collate data from ranks here

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;


    // 打印值 主要是为了check结果是否正确
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, &cells_soa, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);

    write_values(params, &cells_soa, obstacles, av_vels);
    finalise(&params, &cells_soa, &tmp_cells_soa, &obstacles, &av_vels);

    return EXIT_SUCCESS;
}

float fusion_more(const t_param params, t_speeds_soa* restrict cells_soa, t_speeds_soa* restrict tmp_cells_soa,
                  const int* restrict obstacles, const float w11,
                  const float w22, const float c_sq, const float w0, const float w1, const float w2,
                  const float divideVal, const float divideVal2){
//    __assume_aligned(cells_soa, 64);
//    __assume_aligned(cells_soa->speeds0, 64);
//    __assume_aligned(cells_soa->speeds1, 64);
//    __assume_aligned(cells_soa->speeds2, 64);
//    __assume_aligned(cells_soa->speeds3, 64);
//    __assume_aligned(cells_soa->speeds4, 64);
//    __assume_aligned(cells_soa->speeds5, 64);
//    __assume_aligned(cells_soa->speeds6, 64);
//    __assume_aligned(cells_soa->speeds7, 64);
//    __assume_aligned(cells_soa->speeds8, 64);
//    __assume_aligned(tmp_cells_soa, 64);
//    __assume_aligned(tmp_cells_soa->speeds0, 64);
//    __assume_aligned(tmp_cells_soa->speeds1, 64);
//    __assume_aligned(tmp_cells_soa->speeds2, 64);
//    __assume_aligned(tmp_cells_soa->speeds3, 64);
//    __assume_aligned(tmp_cells_soa->speeds4, 64);
//    __assume_aligned(tmp_cells_soa->speeds5, 64);
//    __assume_aligned(tmp_cells_soa->speeds6, 64);
//    __assume_aligned(tmp_cells_soa->speeds7, 64);
//    __assume_aligned(tmp_cells_soa->speeds8, 64);
//    __assume_aligned(obstacles, 64);

    int tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */
    /* initialise */
    tot_u = 0.f;

    /* modify the 2nd row of the grid */
    // accelerate flow!!!!!
    const int secondRow = params.ny - 2;

//    __assume(params.nx % 2 == 0);
//    __assume(params.nx % 4 == 0);
//    __assume(params.nx % 8 == 0);
//    __assume(params.nx % 16 == 0);
//    __assume(params.nx % 32 == 0);
//    __assume(params.nx % 64 == 0);
//    __assume(params.nx % 128 == 0);

    // 给倒数第二行的每个cell加上w11和w22
#pragma omp simd
    for (int ii = 0; ii < params.nx; ii++) {
        /* if the cell is not occupied and
        ** we don't send a negative density */
        const int idx = ii + secondRow * params.nx;

        if (!obstacles[idx]
            && (cells_soa->speeds3[idx] - w11) > 0.f
            && (cells_soa->speeds6[idx] - w22) > 0.f
            && (cells_soa->speeds7[idx] - w22) > 0.f) {
            cells_soa->speeds1[idx] += w11;
            cells_soa->speeds5[idx] += w22;
            cells_soa->speeds8[idx] += w22;
            cells_soa->speeds3[idx] -= w11;
            cells_soa->speeds6[idx] -= w22;
            cells_soa->speeds7[idx] -= w22;
        }
    }

    //all above is accelerate_flow()!!!!!
    /* loop over _all_ cells */

//    __assume(params.ny % 2 == 0);
//    __assume(params.ny % 4 == 0);
//    __assume(params.ny % 8 == 0);
//    __assume(params.ny % 16 == 0);
//    __assume(params.ny % 32 == 0);
//    __assume(params.ny % 64 == 0);
//    __assume(params.ny % 128 == 0);

#pragma omp parallel for collapse(2) reduction(+:tot_u, tot_cells) schedule(static)
    for (int jj = 0; jj < params.ny; jj++) {
        for (int ii = 0; ii < params.nx; ii++) {
            // propagate()!!!!
            /* determine indices of axis-direction neighbours
            ** respecting periodic boundary conditions (wrap around) */
            const int y_n = (jj + 1) % params.ny;
            const int x_e = (ii + 1) % params.nx;
            const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
            const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */

            float keep[9]__attribute__((aligned(64)));
            keep[0] = cells_soa->speeds0[ii + jj * params.nx];
            keep[1] = cells_soa->speeds1[x_w + jj * params.nx];
            keep[2] = cells_soa->speeds2[ii + y_s * params.nx];
            keep[3] = cells_soa->speeds3[x_e + jj * params.nx];
            keep[4] = cells_soa->speeds4[ii + y_n * params.nx];
            keep[5] = cells_soa->speeds5[x_w + y_s * params.nx];
            keep[6] = cells_soa->speeds6[x_e + y_s * params.nx];
            keep[7] = cells_soa->speeds7[x_e + y_n * params.nx];
            keep[8] = cells_soa->speeds8[x_w + y_n * params.nx];
            /* if the cell does not contains an obstacle */
            if (!obstacles[jj * params.nx + ii]) {
                //collision()!!!!!!!!!!!!
                /* compute local density total */
                float local_density = 0.f;

                local_density = keep[0]
                                + keep[1]
                                + keep[2]
                                + keep[3]
                                + keep[4]
                                + keep[5]
                                + keep[6]
                                + keep[7]
                                + keep[8];

                /* compute x velocity component */
                float u_x = (keep[1]
                             + keep[5]
                             + keep[8]
                             - (keep[3]
                                + keep[6]
                                + keep[7]))
                            / local_density;
                /* compute y velocity component */
                float u_y = (keep[2]
                             + keep[5]
                             + keep[6]
                             - (keep[4]
                                + keep[7]
                                + keep[8]))
                            / local_density;

                /* velocity squared */
                const float u_sq = u_x * u_x + u_y * u_y;
                const float w1Local = w1 * local_density;
                const float w2Local = w2 * local_density;
                const float val = u_sq / divideVal2;

                /* directional velocity components */
                float u[NSPEEDS];
                u[1] = u_x;        /* east */
                u[2] = u_y;  /* north */
                u[3] = -u_x;        /* west */
                u[4] = -u_y;  /* south */
                u[5] = u_x + u_y;  /* north-east */
                u[6] = -u_x + u_y;  /* north-west */
                u[7] = -u_x - u_y;  /* south-west */
                u[8] = u_x - u_y;  /* south-east */

                /* equilibrium densities */
                float d_equ[NSPEEDS];
                /* zero velocity density: weight w0 */
                d_equ[0] = w0 * local_density
                           * (1.f - u_sq / (2.f * c_sq));
                d_equ[1] = w1Local * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / divideVal
                                      - val);
                d_equ[2] = w1Local * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / divideVal
                                      - val);
                d_equ[3] = w1Local * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / divideVal
                                      - val);
                d_equ[4] = w1Local * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / divideVal
                                      - val);
                /* diagonal speeds: weight w2 */
                d_equ[5] = w2Local * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / divideVal
                                      - val);
                d_equ[6] = w2Local * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / divideVal
                                      - val);
                d_equ[7] = w2Local * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / divideVal
                                      - val);
                d_equ[8] = w2Local * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / divideVal
                                      - val);

                /* relaxation step */
                tmp_cells_soa->speeds0[ii + jj * params.nx] = keep[0] + params.omega * (d_equ[0] - keep[0]);
                tmp_cells_soa->speeds1[ii + jj * params.nx] = keep[1] + params.omega * (d_equ[1] - keep[1]);
                tmp_cells_soa->speeds2[ii + jj * params.nx] = keep[2] + params.omega * (d_equ[2] - keep[2]);
                tmp_cells_soa->speeds3[ii + jj * params.nx] = keep[3] + params.omega * (d_equ[3] - keep[3]);
                tmp_cells_soa->speeds4[ii + jj * params.nx] = keep[4] + params.omega * (d_equ[4] - keep[4]);
                tmp_cells_soa->speeds5[ii + jj * params.nx] = keep[5] + params.omega * (d_equ[5] - keep[5]);
                tmp_cells_soa->speeds6[ii + jj * params.nx] = keep[6] + params.omega * (d_equ[6] - keep[6]);
                tmp_cells_soa->speeds7[ii + jj * params.nx] = keep[7] + params.omega * (d_equ[7] - keep[7]);
                tmp_cells_soa->speeds8[ii + jj * params.nx] = keep[8] + params.omega * (d_equ[8] - keep[8]);

                // av_velocity()!!!
                /* local density total */
                local_density = tmp_cells_soa->speeds0[ii + jj * params.nx]
                                + tmp_cells_soa->speeds1[ii + jj * params.nx]
                                + tmp_cells_soa->speeds2[ii + jj * params.nx]
                                + tmp_cells_soa->speeds3[ii + jj * params.nx]
                                + tmp_cells_soa->speeds4[ii + jj * params.nx]
                                + tmp_cells_soa->speeds5[ii + jj * params.nx]
                                + tmp_cells_soa->speeds6[ii + jj * params.nx]
                                + tmp_cells_soa->speeds7[ii + jj * params.nx]
                                + tmp_cells_soa->speeds8[ii + jj * params.nx];

                u_x = (tmp_cells_soa->speeds1[ii + jj * params.nx]
                       + tmp_cells_soa->speeds5[ii + jj * params.nx]
                       + tmp_cells_soa->speeds8[ii + jj * params.nx]
                       - (tmp_cells_soa->speeds3[ii + jj * params.nx]
                          + tmp_cells_soa->speeds6[ii + jj * params.nx]
                          + tmp_cells_soa->speeds7[ii + jj * params.nx]))
                      / local_density;
                u_y = (tmp_cells_soa->speeds2[ii + jj * params.nx]
                       + tmp_cells_soa->speeds5[ii + jj * params.nx]
                       + tmp_cells_soa->speeds6[ii + jj * params.nx]
                       - (tmp_cells_soa->speeds4[ii + jj * params.nx]
                          + tmp_cells_soa->speeds7[ii + jj * params.nx]
                          + tmp_cells_soa->speeds8[ii + jj * params.nx]))
                      / local_density;
                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                /* increase counter of inspected cells */
                ++tot_cells;
            } else {
                //rebound()!!!!!!!!!!!
                /* called after propagate, so taking values from scratch space
                ** mirroring, and writing into main grid */
                tmp_cells_soa->speeds0[ii + jj * params.nx] = keep[0];
                tmp_cells_soa->speeds1[ii + jj * params.nx] = keep[3];
                tmp_cells_soa->speeds2[ii + jj * params.nx] = keep[4];
                tmp_cells_soa->speeds3[ii + jj * params.nx] = keep[1];
                tmp_cells_soa->speeds4[ii + jj * params.nx] = keep[2];
                tmp_cells_soa->speeds5[ii + jj * params.nx] = keep[7];
                tmp_cells_soa->speeds6[ii + jj * params.nx] = keep[8];
                tmp_cells_soa->speeds7[ii + jj * params.nx] = keep[5];
                tmp_cells_soa->speeds8[ii + jj * params.nx] = keep[6];
            }
        }
    }

    return tot_u / (float)tot_cells;
}

// 这个函数主要是用在计算雷诺平均里面的，是为了check最后结果的，不是很重要
float av_velocity(const t_param params, t_speeds_soa* cells_soa, const int* obstacles)
{
//    __assume_aligned(cells_soa, 64);
//    __assume_aligned(cells_soa->speeds0, 64);
//    __assume_aligned(cells_soa->speeds1, 64);
//    __assume_aligned(cells_soa->speeds2, 64);
//    __assume_aligned(cells_soa->speeds3, 64);
//    __assume_aligned(cells_soa->speeds4, 64);
//    __assume_aligned(cells_soa->speeds5, 64);
//    __assume_aligned(cells_soa->speeds6, 64);
//    __assume_aligned(cells_soa->speeds7, 64);
//    __assume_aligned(cells_soa->speeds8, 64);
//    __assume_aligned(obstacles, 64);
//
//    __assume(params.nx % 2 == 0);
//    __assume(params.nx % 4 == 0);
//    __assume(params.nx % 8 == 0);
//    __assume(params.nx % 16 == 0);
//    __assume(params.nx % 32 == 0);
//    __assume(params.nx % 64 == 0);
//    __assume(params.nx % 128 == 0);
//
//    __assume(params.ny % 2 == 0);
//    __assume(params.ny % 4 == 0);
//    __assume(params.ny % 8 == 0);
//    __assume(params.ny % 16 == 0);
//    __assume(params.ny % 32 == 0);
//    __assume(params.ny % 64 == 0);
//    __assume(params.ny % 128 == 0);
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;

    /* loop over all non-blocked cells */
#pragma omp parallel for collapse(2) reduction(+:tot_u, tot_cells) schedule(static)
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* ignore occupied cells */
            if (!obstacles[ii + jj*params.nx])
            {
                /* local density total */
                float local_density = 0.f;

                local_density = cells_soa->speeds0[ii + jj*params.nx]
                                + cells_soa->speeds1[ii + jj*params.nx]
                                + cells_soa->speeds2[ii + jj*params.nx]
                                + cells_soa->speeds3[ii + jj*params.nx]
                                + cells_soa->speeds4[ii + jj*params.nx]
                                + cells_soa->speeds5[ii + jj*params.nx]
                                + cells_soa->speeds6[ii + jj*params.nx]
                                + cells_soa->speeds7[ii + jj*params.nx]
                                + cells_soa->speeds8[ii + jj*params.nx];

                /* x-component of velocity */
                float u_x = (cells_soa->speeds1[ii + jj*params.nx]
                             + cells_soa->speeds5[ii + jj*params.nx]
                             + cells_soa->speeds8[ii + jj*params.nx]
                             - (cells_soa->speeds3[ii + jj*params.nx]
                                + cells_soa->speeds6[ii + jj*params.nx]
                                + cells_soa->speeds7[ii + jj*params.nx]))
                            / local_density;

                /* compute y velocity component */
                float u_y = (cells_soa->speeds2[ii + jj*params.nx]
                             + cells_soa->speeds5[ii + jj*params.nx]
                             + cells_soa->speeds6[ii + jj*params.nx]
                             - (cells_soa->speeds4[ii + jj*params.nx]
                                + cells_soa->speeds7[ii + jj*params.nx]
                                + cells_soa->speeds8[ii + jj*params.nx]))
                            / local_density;
                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }

    return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speeds_soa* cells_soa, t_speeds_soa* tmp_cells_soa,
               int** obstacles_ptr, float** av_vels_ptr) {
    char message[1024];  /* message buffer */
    FILE *fp;            /* file pointer */
    int xx, yy;         /* generic array indices */
    int blocked;        /* indicates whether a cell is blocked by an obstacle */
    int retval;         /* to hold return value for checking */

    /* open the parameter file */
    fp = fopen(paramfile, "r");

    if (fp == NULL) {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));

    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->ny));

    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->maxIters));

    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

    if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->density));

    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->accel));

    if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->omega));

    if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    /*
    ** Allocate memory.
    **
    ** Remember C is pass-by-value, so we need to
    ** pass pointers into the initialise function.
    **
    ** NB we are allocating a 1D array, so that the
    ** memory will be contiguous.  We still want to
    ** index this memory as if it were a (row major
    ** ordered) 2D array, however.  We will perform
    ** some arithmetic using the row and column
    ** coordinates, inside the square brackets, when
    ** we want to access elements of this array.
    **
    ** Note also that we are using a structure to
    ** hold an array of 'speeds'.  We will allocate
    ** a 1D array of these structs.
    */

    /* main grid */

    cells_soa->speeds0 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    cells_soa->speeds1 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    cells_soa->speeds2 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    cells_soa->speeds3 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    cells_soa->speeds4 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    cells_soa->speeds5 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    cells_soa->speeds6 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    cells_soa->speeds7 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    cells_soa->speeds8 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);

    /* 'helper' grid, used as scratch space */

    tmp_cells_soa->speeds0 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    tmp_cells_soa->speeds1 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    tmp_cells_soa->speeds2 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    tmp_cells_soa->speeds3 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    tmp_cells_soa->speeds4 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    tmp_cells_soa->speeds5 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    tmp_cells_soa->speeds6 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    tmp_cells_soa->speeds7 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);
    tmp_cells_soa->speeds8 = (float *) _mm_malloc(params->ny * params->nx * sizeof(float), 64);

    /* the map of obstacles */
    *obstacles_ptr = (int *) _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

//    __assume_aligned(cells_soa, 64);
//    __assume_aligned(cells_soa->speeds0, 64);
//    __assume_aligned(cells_soa->speeds1, 64);
//    __assume_aligned(cells_soa->speeds2, 64);
//    __assume_aligned(cells_soa->speeds3, 64);
//    __assume_aligned(cells_soa->speeds4, 64);
//    __assume_aligned(cells_soa->speeds5, 64);
//    __assume_aligned(cells_soa->speeds6, 64);
//    __assume_aligned(cells_soa->speeds7, 64);
//    __assume_aligned(cells_soa->speeds8, 64);
//    __assume_aligned(tmp_cells_soa, 64);
//    __assume_aligned(tmp_cells_soa->speeds0, 64);
//    __assume_aligned(tmp_cells_soa->speeds1, 64);
//    __assume_aligned(tmp_cells_soa->speeds2, 64);
//    __assume_aligned(tmp_cells_soa->speeds3, 64);
//    __assume_aligned(tmp_cells_soa->speeds4, 64);
//    __assume_aligned(tmp_cells_soa->speeds5, 64);
//    __assume_aligned(tmp_cells_soa->speeds6, 64);
//    __assume_aligned(tmp_cells_soa->speeds7, 64);
//    __assume_aligned(tmp_cells_soa->speeds8, 64);
//    __assume_aligned(*obstacles_ptr, 64);

    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density / 9.f;
    float w2 = params->density / 36.f;

#pragma omp parallel for collapse(2) schedule(static)
    for (int jj = 0; jj < params->ny; jj++) {
        for (int ii = 0; ii < params->nx; ii++) {
            /* centre */
            cells_soa->speeds0[ii + jj * params->nx] = w0;
            /* axis directions */
            cells_soa->speeds1[ii + jj * params->nx] = w1;
            cells_soa->speeds2[ii + jj * params->nx] = w1;
            cells_soa->speeds3[ii + jj * params->nx] = w1;
            cells_soa->speeds4[ii + jj * params->nx] = w1;
            /* diagonals */
            cells_soa->speeds5[ii + jj * params->nx] = w2;
            cells_soa->speeds6[ii + jj * params->nx] = w2;
            cells_soa->speeds7[ii + jj * params->nx] = w2;
            cells_soa->speeds8[ii + jj * params->nx] = w2;

            /* initialise obstacle map */
            (*obstacles_ptr)[ii + jj * params->nx] = 0;
        }
    }


    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");

    if (fp == NULL) {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* assign to array */
        (*obstacles_ptr)[xx + yy * params->nx] = blocked;
    }

    /* and close the file */
    fclose(fp);

    /*
    ** allocate space to hold a record of the avarage velocities computed
    ** at each timestep
    */
    *av_vels_ptr = (float *) _mm_malloc(sizeof(float) * params->maxIters, 64);

    return EXIT_SUCCESS;

}

int finalise(const t_param* params, t_speeds_soa* cells_soa, t_speeds_soa* tmp_cells_soa,
             int** obstacles_ptr, float** av_vels_ptr) {

    _mm_free(cells_soa->speeds0);
    _mm_free(cells_soa->speeds1);
    _mm_free(cells_soa->speeds2);
    _mm_free(cells_soa->speeds3);
    _mm_free(cells_soa->speeds4);
    _mm_free(cells_soa->speeds5);
    _mm_free(cells_soa->speeds6);
    _mm_free(cells_soa->speeds7);
    _mm_free(cells_soa->speeds8);

    _mm_free(tmp_cells_soa->speeds0);
    _mm_free(tmp_cells_soa->speeds1);
    _mm_free(tmp_cells_soa->speeds2);
    _mm_free(tmp_cells_soa->speeds3);
    _mm_free(tmp_cells_soa->speeds4);
    _mm_free(tmp_cells_soa->speeds5);
    _mm_free(tmp_cells_soa->speeds6);
    _mm_free(tmp_cells_soa->speeds7);
    _mm_free(tmp_cells_soa->speeds8);

    _mm_free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    _mm_free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    return EXIT_SUCCESS;
}

// 计算雷诺平均，不重要，主要是为了比较结果是否正确
float calc_reynolds(const t_param params, t_speeds_soa* cells_soa, int* obstacles)
{
    const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

    return av_velocity(params, cells_soa, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speeds_soa* cells_soa)
{
    float total = 0.f;  /* accumulator */

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            total += cells_soa->speeds0[ii + jj*params.nx]
                     + cells_soa->speeds1[ii + jj*params.nx]
                     + cells_soa->speeds2[ii + jj*params.nx]
                     + cells_soa->speeds3[ii + jj*params.nx]
                     + cells_soa->speeds4[ii + jj*params.nx]
                     + cells_soa->speeds5[ii + jj*params.nx]
                     + cells_soa->speeds6[ii + jj*params.nx]
                     + cells_soa->speeds7[ii + jj*params.nx]
                     + cells_soa->speeds8[ii + jj*params.nx];
        }
    }

    return total;
}

int write_values(const t_param params, t_speeds_soa* cells_soa, int* obstacles, float* av_vels)
{
    FILE* fp;                     /* file pointer */
    const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
    float local_density;         /* per grid cell sum of densities */
    float pressure;              /* fluid pressure in grid cell */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(FINALSTATEFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* an occupied cell */
            if (obstacles[ii + jj*params.nx])
            {
                u_x = u_y = u = 0.f;
                pressure = params.density * c_sq;
            }
                /* no obstacle */
            else
            {
                local_density = 0.f;

                local_density = cells_soa->speeds0[ii + jj*params.nx]
                                + cells_soa->speeds1[ii + jj*params.nx]
                                + cells_soa->speeds2[ii + jj*params.nx]
                                + cells_soa->speeds3[ii + jj*params.nx]
                                + cells_soa->speeds4[ii + jj*params.nx]
                                + cells_soa->speeds5[ii + jj*params.nx]
                                + cells_soa->speeds6[ii + jj*params.nx]
                                + cells_soa->speeds7[ii + jj*params.nx]
                                + cells_soa->speeds8[ii + jj*params.nx];

                u_x = (cells_soa->speeds1[ii + jj*params.nx]
                       + cells_soa->speeds5[ii + jj*params.nx]
                       + cells_soa->speeds8[ii + jj*params.nx]
                       - (cells_soa->speeds3[ii + jj*params.nx]
                          + cells_soa->speeds6[ii + jj*params.nx]
                          + cells_soa->speeds7[ii + jj*params.nx]))
                      / local_density;

                u_y = (cells_soa->speeds2[ii + jj*params.nx]
                       + cells_soa->speeds5[ii + jj*params.nx]
                       + cells_soa->speeds6[ii + jj*params.nx]
                       - (cells_soa->speeds4[ii + jj*params.nx]
                          + cells_soa->speeds7[ii + jj*params.nx]
                          + cells_soa->speeds8[ii + jj*params.nx]))
                      / local_density;
                /* compute norm of velocity */
                u = sqrtf((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
        }
    }

    fclose(fp);

    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int ii = 0; ii < params.maxIters; ii++)
    {
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);

    return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}
