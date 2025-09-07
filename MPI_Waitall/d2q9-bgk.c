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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>
#include <string.h>

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

/* struct to hold the 'speed' values */
typedef struct
{
    float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,t_speed** results_ptr,
               int** obstacles_ptr, float** av_vels_ptr,int ThisRank, int ProcessSize,int* KeepTotalRows,
               int** obstacles_ptr_Total, int* numberOfNonObstacles);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
//float fusion_more(const t_param params, t_speed* cells, t_speed* tmp_cells, const int* obstacles, float w11, float w22,
//                  float c_sq, float w0, float w1, float w2, float divideVal, float divideVal2, int ThisRank,int ProcessSize);

float fusion_more(const t_param params, t_speed* cells, t_speed* tmp_cells,
                  const int* obstacles, float w11, float w22,
                  float c_sq, float w0, float w1, float w2,
                  float divideVal, float divideVal2,
                  int ThisRank, int ProcessSize,
                  int rowStart, int rowEnd,int do_accel);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_speed** results_ptr, int** obstacles_ptr_Total);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, const int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);


void write_animation_data_mpi(const t_param params, t_speed* cells, const int timestep,
                              const int* obstacles, int ThisRank, int ProcessSize,
                              int KeepTotalRows, const int* obstacles_Total,
                              MPI_Datatype MPI_T_SPEED);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{

    int ThisRank, ProcessSize;

    MPI_Init(&argc, &argv);
    // Get the number of processes and current process rank in MPI environment
    // and store them in ThisRank and ProcessSize
    MPI_Comm_size(MPI_COMM_WORLD, &ProcessSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisRank);


    // Assume t_speed struct and params have been defined and initialized

    // Make a t_speed struct be sent as 9 consecutive floats in communication, convenient for halo exchange, convergence, etc.
    MPI_Datatype MPI_T_SPEED;
    MPI_Type_contiguous(9, MPI_FLOAT, &MPI_T_SPEED);
    MPI_Type_commit(&MPI_T_SPEED);


    // Print process information
    printf("Process %d of %d started.\n", ThisRank, ProcessSize);

    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    t_param  params;              /* struct to hold parameter values */
    t_speed* cells     = NULL;    /* grid containing fluid densities */
    t_speed* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
    struct timeval timstr;                                                             /* structure to hold elapsed time */
    double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

    t_speed* results = NULL;      /* results */
    int KeepTotalRows = 0;
    int* obstacles_Total = NULL;
    int numberOfNonObstacles = 0;

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

    // Initialize here, initialize a results array, only used when rank=0, for final data integration
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells ,&results, &obstacles, &av_vels, ThisRank,ProcessSize,&KeepTotalRows
            ,&obstacles_Total, &numberOfNonObstacles);

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

    // This is the total number of elements each process needs to handle
    int local_n_Size = (params.ny-2)*params.nx;

    for(int tt = 0; tt < params.maxIters; tt++)
    {
        int up_neighbor = (ThisRank - 1 + ProcessSize) % ProcessSize;
        int down_neighbor = (ThisRank + 1) % ProcessSize;

        // *** MOD ②: Non-blocking communication + computation overlap
        // req is the returned handle, used for subsequent querying or waiting for this non-blocking operation to complete.
        MPI_Request req[4];

        // 10 and 11 are tags, integer labels used to distinguish different messages. Must match on both send and receive ends.
        // (A) post non-blocking halo exchange
        // MPI_Isend first parameter is the starting address of send buffer (skip first halo row)
        MPI_Isend(cells+params.nx,              params.nx, MPI_T_SPEED, up_neighbor,   11, MPI_COMM_WORLD, &req[2]);             // Send first real row
        MPI_Isend(cells+(params.ny-2)*params.nx,params.nx, MPI_T_SPEED, down_neighbor, 10, MPI_COMM_WORLD, &req[3]);             // Send last real row

        // MPI_Irecv first parameter is the starting address of receive buffer
        MPI_Irecv(cells,                        params.nx, MPI_T_SPEED, up_neighbor,   10, MPI_COMM_WORLD, &req[0]);             // Upper halo
        MPI_Irecv(cells+(params.ny-1)*params.nx,params.nx, MPI_T_SPEED, down_neighbor, 11, MPI_COMM_WORLD, &req[1]);             // Lower halo

        // (B) First compute internal rows (2 .. ny-3), can be done in parallel with network transmission
        // This do_accel parameter is important to prevent duplicate acceleration; for the last process, only need to accelerate once here, no need to accelerate in subsequent halo-dependent parts
        float tot_u_in  = fusion_more(params, cells, tmp_cells, obstacles,
                                      w11, w22, c_sq, w0, w1, w2,
                                      divideVal, divideVal2,
                                      ThisRank, ProcessSize,
                /*rowStart*/2, /*rowEnd*/params.ny-3, 1);

        // (C) Wait for halo completion
        // MPI_Waitall(4, req, MPI_STATUSES_IGNORE); blocks the current thread
        // until all 4 non-blocking communications represented by req[0]...req[3] are completed.
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        // (D) Then compute halo-dependent boundary rows 1 and ny-2
        float tot_u_bd  = fusion_more(params, cells, tmp_cells, obstacles,
                                      w11, w22, c_sq, w0, w1, w2,
                                      divideVal, divideVal2,
                                      ThisRank, ProcessSize, 1, 1, 0);
        tot_u_bd +=      fusion_more(params, cells, tmp_cells, obstacles,
                                     w11, w22, c_sq, w0, w1, w2,
                                     divideVal, divideVal2,
                                     ThisRank, ProcessSize, params.ny-2, params.ny-2, 0);

        // (E) Sum up this round's velocity and swap pointers
        av_vels[tt] = tot_u_in + tot_u_bd;

        // Output animation data every 100 timesteps
//        if (tt % 100 == 0) {
//            write_animation_data_mpi(params, tmp_cells, tt, obstacles, ThisRank, ProcessSize,
//                                     KeepTotalRows, obstacles_Total, MPI_T_SPEED);
//        }

        t_speed* temp = cells;   // pointer swap
        cells          = tmp_cells;
        tmp_cells      = temp;
    }

    /* Compute time stops here, collate time starts*/
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic=comp_toc;

    // Collate data from ranks here

    // If the results below are different from above, it means there's a problem with copying!

    // Main process collects data here, note that each cells only has a portion of real data, with two halo rows
    // Collect cells and put them in results
    // I think obstacles don't need to be collected, main process keeps a large obstacles
    // main process collect data here, and the operation is blocking operation

    // This part can be implemented with MPI_Gatherv, which allows each process to send different amounts of data, eliminating the need to calculate data volume for each process
    if (ThisRank==0) {
        // Receive cells data from each process
        // Save in results
        // First copy own data to results, results size is the normal large grid size!
        for (int i = 1; i < params.ny - 1; i++) {
            for (int j = 0; j < params.nx; j++) {
                results[j + (i-1) * params.nx] = cells[j + i * params.nx];
            }
        }
        int basic_work = (KeepTotalRows - 3) / ProcessSize; // Minimum number of rows each process handles
        int remainder = (KeepTotalRows - 3) % ProcessSize; // Remainder rows

        // Initialize offsetStart to the size main process handles
        int offsetStart = local_n_Size;

        // Total 112 processes
        for (int i = 1; i < ProcessSize; i++) {
            // Recalculate the number of rows and total elements each process needs to handle
            // Calculate rows for each process
            int rows_per_process = basic_work + (i < remainder ? 1 : 0);
            if (i == ProcessSize - 1) {
                rows_per_process += 3;
            }
            int SizeForEachProcess = rows_per_process * params.nx;

            MPI_Recv(results + offsetStart, SizeForEachProcess, MPI_T_SPEED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offsetStart += SizeForEachProcess;
        }
    }else{
        MPI_Send(cells+params.nx, local_n_Size, MPI_T_SPEED, 0, 0, MPI_COMM_WORLD);
    }

    // This is to calculate the average value for each round
    float total_av_vels[params.maxIters];
    // This line of code makes all processes perform "sum" operation on floating point values in their av_vels array by element position,
    // and aggregates the results to the total_av_vels array on process with rank 0. Specific meaning:
    // total_avels: Only writes reduction results when process rank is 0, other processes can ignore this parameter
    MPI_Reduce(av_vels, total_av_vels, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ThisRank == 0){
        // Calculate average value for each round
        for (int tt = 0; tt < params.maxIters; tt++){
            av_vels[tt] = total_av_vels[tt] / (float)numberOfNonObstacles;
        }
    }

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;
//    /* write final values and free memory */
// Write data in main process, then make check should verify the data written here.
    if (ThisRank == 0){
        // Assign correct params->ny back before writing
        params.ny = KeepTotalRows;
        printf("==done==\n");
        printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, results, obstacles_Total));
        printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
        printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
        printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
        printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
        write_values(params, results, obstacles_Total, av_vels);
    }
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &results, &obstacles_Total);
    MPI_Finalize();
    return EXIT_SUCCESS;
}

float fusion_more(const t_param params, t_speed* cells, t_speed* tmp_cells,const int* obstacles, const float w11,const float w22,
                  const float c_sq, const float w0,const float w1,const float w2,
                  const float divideVal,const float divideVal2, int ThisRank, int ProcessSize ,
                  int rowStart, int rowEnd, int do_accel){

    /* modify the 2nd row of the grid */
    // accelerate flow!!!!!
    // if this is the last process
    // Only perform acceleration operation for the last process
    // If it's the last process, perform acceleration operation on the second-to-last row
    if (do_accel==1 && ThisRank == ProcessSize-1 ){
        const int LastSecondRow = params.ny - 3;
        // LastSecondRow = 3
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* if the cell is not occupied and
            ** we don't send a negative density */

            // Note the obstacle index here
            if (!obstacles[ii + (LastSecondRow-1)*params.nx]
                && (cells[ii + LastSecondRow*params.nx].speeds[3] - w11) > 0.f
                && (cells[ii + LastSecondRow*params.nx].speeds[6] - w22) > 0.f
                && (cells[ii + LastSecondRow*params.nx].speeds[7] - w22) > 0.f)
            {
                /* increase 'east-side' densities */
                cells[ii + LastSecondRow*params.nx].speeds[1] += w11;
                cells[ii + LastSecondRow*params.nx].speeds[5] += w22;
                cells[ii + LastSecondRow*params.nx].speeds[8] += w22;
                /* decrease 'west-side' densities */
                cells[ii + LastSecondRow*params.nx].speeds[3] -= w11;
                cells[ii + LastSecondRow*params.nx].speeds[6] -= w22;
                cells[ii + LastSecondRow*params.nx].speeds[7] -= w22;
            }
        }
    }

    // All above is accelerate_flow()!!!!!
    float tot_u;          /* accumulated magnitudes of velocity for each cell */
    /* initialise */
    tot_u = 0.f;

    /* loop over _all_ cells */
    // jj index from 1 to params.ny-2
    // ii index from 0 to params.nx-1
    // Because jj=0 and last row are halo parts from other processes, this process handles data from 1 to params.ny-2
    for (int jj = rowStart; jj <= rowEnd; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            // propagate()!!!!
            /* determine indices of axis-direction neighbours
            ** respecting periodic boundary conditions (wrap around) */
            const int y_n = (jj + 1) % params.ny;
            const int x_e = (ii + 1) % params.nx;
            const int y_s = jj - 1;
            const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */

            float keep[9];
            keep[0]=cells[ii + jj*params.nx].speeds[0];
            keep[1]=cells[x_w + jj*params.nx].speeds[1];
            keep[2]=cells[ii + y_s*params.nx].speeds[2];
            keep[3]=cells[x_e + jj*params.nx].speeds[3];
            keep[4]=cells[ii + y_n*params.nx].speeds[4];
            keep[5]=cells[x_w + y_s*params.nx].speeds[5];
            keep[6]=cells[x_e + y_s*params.nx].speeds[6];
            keep[7]=cells[x_e + y_n*params.nx].speeds[7];
            keep[8]=cells[x_w + y_n*params.nx].speeds[8];

            /* if the cell contains an obstacle */
            if (!obstacles[(jj-1)*params.nx + ii])
            {
                //collision()!!!!!!!!!!!!
                /* compute local density total */
                float local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += keep[kk];
                }

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

                /* directional velocity components */
                float u[NSPEEDS];
                u[1] =   u_x;        /* east */
                u[2] =   u_y;  /* north */
                u[3] = - u_x;        /* west */
                u[4] = - u_y;  /* south */
                u[5] =   u_x + u_y;  /* north-east */
                u[6] = - u_x + u_y;  /* north-west */
                u[7] = - u_x - u_y;  /* south-west */
                u[8] =   u_x - u_y;  /* south-east */

                /* equilibrium densities */
                float d_equ[NSPEEDS];
                /* zero velocity density: weight w0 */
                d_equ[0] = w0 * local_density
                           * (1.f - u_sq / (2.f * c_sq));

                const float w1Local = w1 * local_density;
                const float w2Local = w2 * local_density;
                const float val = u_sq / divideVal2;
                /* axis speeds: weight w1 */
                d_equ[1] = w1Local * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / divideVal
                                      - val );
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
                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    tmp_cells[ii + jj*params.nx].speeds[kk] = keep[kk]
                                                              + params.omega
                                                                * (d_equ[kk] - keep[kk]);
                }

                // av_velocity()!!!
                /* local density total */
                local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
                }

                /* x-component of velocity */
                u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                       + tmp_cells[ii + jj*params.nx].speeds[5]
                       + tmp_cells[ii + jj*params.nx].speeds[8]
                       - (tmp_cells[ii + jj*params.nx].speeds[3]
                          + tmp_cells[ii + jj*params.nx].speeds[6]
                          + tmp_cells[ii + jj*params.nx].speeds[7]))
                      / local_density;
                /* compute y velocity component */
                u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                       + tmp_cells[ii + jj*params.nx].speeds[5]
                       + tmp_cells[ii + jj*params.nx].speeds[6]
                       - (tmp_cells[ii + jj*params.nx].speeds[4]
                          + tmp_cells[ii + jj*params.nx].speeds[7]
                          + tmp_cells[ii + jj*params.nx].speeds[8]))
                      / local_density;
                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
            }else{
                //rebound()!!!!!!!!!!!
                /* called after propagate, so taking values from scratch space
                ** mirroring, and writing into main grid */

                // Note that if it's rebound, its speeds[0] remains unchanged
                tmp_cells[ii + jj*params.nx].speeds[1] = keep[3];
                tmp_cells[ii + jj*params.nx].speeds[2] = keep[4];
                tmp_cells[ii + jj*params.nx].speeds[3] = keep[1];
                tmp_cells[ii + jj*params.nx].speeds[4] = keep[2];
                tmp_cells[ii + jj*params.nx].speeds[5] = keep[7];
                tmp_cells[ii + jj*params.nx].speeds[6] = keep[8];
                tmp_cells[ii + jj*params.nx].speeds[7] = keep[5];
                tmp_cells[ii + jj*params.nx].speeds[8] = keep[6];
            }
        }
    }
    return tot_u;
}

float av_velocity(const t_param params, t_speed* cells, const int* obstacles)
{
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;

    /* loop over all non-blocked cells */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* ignore occupied cells */
            if (!obstacles[ii + jj*params.nx])
            {
                /* local density total */
                float local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[ii + jj*params.nx].speeds[kk];
                }

                /* x-component of velocity */
                float u_x = (cells[ii + jj*params.nx].speeds[1]
                             + cells[ii + jj*params.nx].speeds[5]
                             + cells[ii + jj*params.nx].speeds[8]
                             - (cells[ii + jj*params.nx].speeds[3]
                                + cells[ii + jj*params.nx].speeds[6]
                                + cells[ii + jj*params.nx].speeds[7]))
                            / local_density;
                /* compute y velocity component */
                float u_y = (cells[ii + jj*params.nx].speeds[2]
                             + cells[ii + jj*params.nx].speeds[5]
                             + cells[ii + jj*params.nx].speeds[6]
                             - (cells[ii + jj*params.nx].speeds[4]
                                + cells[ii + jj*params.nx].speeds[7]
                                + cells[ii + jj*params.nx].speeds[8]))
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
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,t_speed** results_ptr,
               int** obstacles_ptr, float** av_vels_ptr,int ThisRank, int ProcessSize, int* KeepTotalRows,
               int** obstacles_ptr_Total, int* numberOfNonObstacles)
{
    char   message[1024];  /* message buffer */
    FILE*   fp;            /* file pointer */
    int    xx, yy;         /* generic array indices */
    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */

    /* open the parameter file */
    fp = fopen(paramfile, "r");

    if (fp == NULL)
    {
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


    // here to keep the total rows
    *KeepTotalRows = params->ny;

    // Main process also handles tasks

    // I need to ensure the last process handles at least 3 rows, which are the last three rows!!!!!
    // This reason doesn't matter much, it's due to acceleration logic - acceleration only affects the second-to-last row of the entire grid, so the last process must handle the last three rows to prevent data conflicts

//    Purpose: Ensure the last process has at least the last 3, 2, 1 rows of the grid to execute accelerate_flow (only accelerates the global second-to-last row).
//
//    Local grid size: local_n = rows_per_process * nx
//
//    Actual computation data rows: 1 … rows_per_process, row 0 and last row are halo.

    int basic_work =  (params->ny-3) / ProcessSize; // Minimum number of rows each process handles
    int remainder = (params->ny-3) % ProcessSize; // Remainder rows
    // If size=8, remainder is at most 7 rows when rows=15, it's not recommended to let the last thread handle 8 rows, can distribute these 7 rows evenly to the first 7 threads
    // This way the first 7 processes each handle 2 rows, and the last process handles 1 row

    // Calculate rows for each process
    int rows_per_process = basic_work + (ThisRank < remainder ? 1 : 0);
    // Last process handles 3 additional rows
    if (ThisRank == ProcessSize-1){
        rows_per_process += 3;
    }

    // Allocate local array size for each process, number of elements a process needs to handle
    // The last process's local_n may differ from other processes
    int local_n = rows_per_process * params->nx;

    // Allocate local array size for each process, and allocate two additional rows to store halo data
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (local_n + 2*params->nx));

    // Similarly allocate local array size for tmp_cells
    // This doesn't need initialization
    *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (local_n + 2*params->nx));

    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    // Initial values are all the same, doesn't matter.
    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density      / 9.f;
    float w2 = params->density      / 36.f;

    // Initialize values for this process, each process initializes the same values, doesn't matter
    // Note, can also initialize upper and lower boundaries to the same values, doesn't matter
    // Because if not initialized, halo exchange will also initialize these two rows
    for (int jj = 0; jj < rows_per_process+2; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            /* centre */
            (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
            (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
            (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
            (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
            (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
            (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
            (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;

        }
    }

    /* the map of obstacles */
    if (ThisRank == 0){
        // initialize results
        // This is used to store results from all processes, main process uses it to collect data from all processes
        *results_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

        // malloc the entire obstacles array
        *obstacles_ptr_Total = (int*)malloc(sizeof(int) * params->nx * params->ny);

        if (*obstacles_ptr_Total == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

        /* first set all cells in obstacle array to zero */
        for (int jj = 0; jj < params->ny; jj++)
        {
            for (int ii = 0; ii < params->nx; ii++)
            {
                (*obstacles_ptr_Total)[ii + jj*params->nx] = 0;
            }
        }

        /* open the obstacle data file */
        fp = fopen(obstaclefile, "r");

        if (fp == NULL)
        {
            sprintf(message, "could not open input obstacles file: %s", obstaclefile);
            die(message, __LINE__, __FILE__);
        }

        /* read-in the blocked cells list */
        while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
        {
            /* some checks */
            if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

            if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

            if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

            if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

            /* assign to array */
            (*obstacles_ptr_Total)[xx + yy*params->nx] = blocked;
        }

        *numberOfNonObstacles = 0;
        // Total number of objects params->ny*params->nx  16384
        for (int jj = 0; jj < params->ny; jj++)
        {
            for (int ii = 0; ii < params->nx; ii++)
            {
                if ((*obstacles_ptr_Total)[ii + jj*params->nx] != 1){
                    (*numberOfNonObstacles)++;
                }
            }
        }

        /* and close the file */
        fclose(fp);

        // At this point, main process has read obstacles, next broadcast (actually send) to other processes, each other process receives obstacles corresponding to their actual cells size to handle
        // Main process also allocates its own obstacles
        // This local_n is the main process's own local_n
        *obstacles_ptr = malloc(sizeof(int) * local_n);
        for (int i = 0; i < local_n; i++) {
            (*obstacles_ptr)[i] = (*obstacles_ptr_Total)[i];
        }
        // Send other obstacles to other processes, each process has different local_n, since this is in main process, need to recalculate size
        for (int i = 1; i < ProcessSize; i++) {
            int rows_for_this_process = basic_work + (i < remainder ? 1 : 0);
            if (i == ProcessSize-1){
                rows_for_this_process += 3;
            }

            // Allocate local array size for each process, number of elements a process needs to handle
            int size_Process = rows_for_this_process * params->nx;

            int start_row = 0;
            // Add up the number of rows each previous process handled to get the starting row for this process
            for (int t = 0; t < i; t++) {
                int rows_for_previous_process = basic_work + (t < remainder ? 1 : 0);
                start_row += rows_for_previous_process;
            }

            // Allocate local array size for each process, no need to allocate halo data
            // Allocate a temporary variable here to send data
            // Actually this part can be omitted, can directly send obstacles_ptr_Total, just calculate the index
            int sendArr[size_Process];
            for (int j = 0; j < size_Process; j++) {
                sendArr[j] = (*obstacles_ptr_Total)[j + (start_row * params->nx)];
            }
            // size_Process here is each process's local_n
            MPI_Send(sendArr, size_Process, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }else{
        // This is other processes
        // Other processes receive obstacles broadcast from main process
        *obstacles_ptr = malloc(sizeof(int) * local_n);
        // printf the size of local_n for this process
        MPI_Recv(*obstacles_ptr, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // here to make sure nx and ny are small fractions

    params->nx = params->nx;
    // The real data to handle is from row 1 to rows_per_process+1
    // the real handle data is from the first row to the rows_per_process+1 row

    // Ensure params->ny is the fragment size here for easier subsequent calculations
    params->ny = rows_per_process+2;

    /*
    ** allocate space to hold a record of the avarage velocities computed
    ** at each timestep
    */
    *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
    return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_speed** results_ptr, int** obstacles_ptr_Total)
{
    /*
    ** free up allocated memory
    */
    free(*cells_ptr);
    *cells_ptr = NULL;

    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;

    free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    free(*results_ptr);
    *results_ptr = NULL;

    free(*obstacles_ptr_Total);
    *obstacles_ptr_Total = NULL;

    return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
    const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

    return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
    float total = 0.f;  /* accumulator */

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            for (int kk = 0; kk < NSPEEDS; kk++)
            {
                total += cells[ii + jj*params.nx].speeds[kk];
            }
        }
    }

    return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
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

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[ii + jj*params.nx].speeds[kk];
                }

                /* compute x velocity component */
                u_x = (cells[ii + jj*params.nx].speeds[1]
                       + cells[ii + jj*params.nx].speeds[5]
                       + cells[ii + jj*params.nx].speeds[8]
                       - (cells[ii + jj*params.nx].speeds[3]
                          + cells[ii + jj*params.nx].speeds[6]
                          + cells[ii + jj*params.nx].speeds[7]))
                      / local_density;
                /* compute y velocity component */
                u_y = (cells[ii + jj*params.nx].speeds[2]
                       + cells[ii + jj*params.nx].speeds[5]
                       + cells[ii + jj*params.nx].speeds[6]
                       - (cells[ii + jj*params.nx].speeds[4]
                          + cells[ii + jj*params.nx].speeds[7]
                          + cells[ii + jj*params.nx].speeds[8]))
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

void write_animation_data_mpi(const t_param params, t_speed* cells, const int timestep,
                              const int* obstacles, int ThisRank, int ProcessSize,
                              int KeepTotalRows, const int* obstacles_Total,
                              MPI_Datatype MPI_T_SPEED) {

    // Create temporary result array to collect data from all processes
    t_speed* temp_results = NULL;

    if (ThisRank == 0) {
        temp_results = (t_speed*)malloc(sizeof(t_speed) * KeepTotalRows * params.nx);
        if (temp_results == NULL) {
            die("cannot allocate memory for temp_results in animation output", __LINE__, __FILE__);
        }
    }

    // Calculate data collection parameters, same logic as at program end
    int basic_work = (KeepTotalRows - 3) / ProcessSize;
    int remainder = (KeepTotalRows - 3) % ProcessSize;

    // Calculate current process data size (excluding halo)
    int rows_per_process = basic_work + (ThisRank < remainder ? 1 : 0);
    if (ThisRank == ProcessSize - 1) {
        rows_per_process += 3;
    }
    int local_data_size = rows_per_process * params.nx;

    // Data collection
    if (ThisRank == 0) {
        // Main process: first copy own data (skip halo rows)
        for (int i = 1; i < params.ny - 1; i++) {
            for (int j = 0; j < params.nx; j++) {
                temp_results[j + (i-1) * params.nx] = cells[j + i * params.nx];
            }
        }

        // Receive data from other processes
        int offsetStart = local_data_size;
        for (int i = 1; i < ProcessSize; i++) {
            int other_rows = basic_work + (i < remainder ? 1 : 0);
            if (i == ProcessSize - 1) {
                other_rows += 3;
            }
            int other_size = other_rows * params.nx;

            MPI_Recv(temp_results + offsetStart, other_size, MPI_T_SPEED, i, 100 + timestep,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offsetStart += other_size;
        }

        // Now main process has complete data, start writing animation file
        char filename[256];
        sprintf(filename, "animation_data/velocity_magnitude_%06d.dat", timestep);

        FILE* fp = fopen(filename, "w");
        if (fp == NULL) {
            die("could not open animation data file", __LINE__, __FILE__);
        }

        // Write grid dimension information
        fprintf(fp, "# nx=%d ny=%d timestep=%d\n", params.nx, KeepTotalRows, timestep);

        // Calculate and write velocity magnitude for each grid point
        for (int jj = 0; jj < KeepTotalRows; jj++) {
            for (int ii = 0; ii < params.nx; ii++) {
                float velocity_magnitude = 0.0f;

                // If not an obstacle, calculate velocity magnitude
                if (!obstacles_Total[ii + jj * params.nx]) {
                    float local_density = 0.f;
                    for (int kk = 0; kk < NSPEEDS; kk++) {
                        local_density += temp_results[ii + jj * params.nx].speeds[kk];
                    }

                    float u_x = (temp_results[ii + jj * params.nx].speeds[1]
                                 + temp_results[ii + jj * params.nx].speeds[5]
                                 + temp_results[ii + jj * params.nx].speeds[8]
                                 - (temp_results[ii + jj * params.nx].speeds[3]
                                    + temp_results[ii + jj * params.nx].speeds[6]
                                    + temp_results[ii + jj * params.nx].speeds[7]))
                                / local_density;

                    float u_y = (temp_results[ii + jj * params.nx].speeds[2]
                                 + temp_results[ii + jj * params.nx].speeds[5]
                                 + temp_results[ii + jj * params.nx].speeds[6]
                                 - (temp_results[ii + jj * params.nx].speeds[4]
                                    + temp_results[ii + jj * params.nx].speeds[7]
                                    + temp_results[ii + jj * params.nx].speeds[8]))
                                / local_density;

                    velocity_magnitude = sqrtf(u_x * u_x + u_y * u_y);
                }

                fprintf(fp, "%.6E\n", velocity_magnitude);
            }
        }

        fclose(fp);
        printf("Written animation data for timestep %d\n", timestep);

        // Free temporary memory
        free(temp_results);

    } else {
        // Other processes: send own data (skip halo rows)
        MPI_Send(cells + params.nx, local_data_size, MPI_T_SPEED, 0, 100 + timestep, MPI_COMM_WORLD);
    }
}
