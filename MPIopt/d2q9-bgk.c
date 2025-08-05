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

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{

    int ThisRank, ProcessSize;

    MPI_Init(&argc, &argv);
    // 在MPI本地环境中获取进程数量和当前进程的rank
    // 并将其存储在 ThisRank 和 ProcessSize 中
    MPI_Comm_size(MPI_COMM_WORLD, &ProcessSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisRank);


    // 假设 t_speed 结构体和 params 已经被定义和初始化

    // 让一个 t_speed 结构体在通信中被当作 9 连续 float 发送，方便 halo 交换、收敛等操作
    MPI_Datatype MPI_T_SPEED;
    MPI_Type_contiguous(9, MPI_FLOAT, &MPI_T_SPEED);
    MPI_Type_commit(&MPI_T_SPEED);


    // 打印一个进程提示信息
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

    // 在这初始化，初始化给一个results数组，这个只有rank=0的时候才用，为了最后整合数据
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

    // 这个是每个进程需要处理的元素总数
    int local_n_Size = (params.ny-2)*params.nx;

    for(int tt = 0; tt < params.maxIters; tt++)
    {
        int up_neighbor = (ThisRank - 1 + ProcessSize) % ProcessSize;
        int down_neighbor = (ThisRank + 1) % ProcessSize;

        // *** MOD ➋: 非阻塞通信 + 计算重叠
        // req是返回的句柄，用于后续查询或等待这条非阻塞操作完成。
        MPI_Request req[4];

        // 10和11是tag 整型标签，用来区分不同消息。必须在收发两端匹配。
        // (A) post non‑blocking halo exchange
        // MPI_Irecv第一个参数是接收缓冲区的起始地址
        MPI_Irecv(cells,                        params.nx, MPI_T_SPEED, up_neighbor,   10, MPI_COMM_WORLD, &req[0]);             // 上 halo
        MPI_Irecv(cells+(params.ny-1)*params.nx,params.nx, MPI_T_SPEED, down_neighbor, 11, MPI_COMM_WORLD, &req[1]);             // 下 halo
        // MPI_Isend第一个参数是发送缓冲区的起始地址（跳过第一行halo行）
        MPI_Isend(cells+params.nx,              params.nx, MPI_T_SPEED, up_neighbor,   11, MPI_COMM_WORLD, &req[2]);             // 发送第一真实行
        MPI_Isend(cells+(params.ny-2)*params.nx,params.nx, MPI_T_SPEED, down_neighbor, 10, MPI_COMM_WORLD, &req[3]);             // 发送最后真实行

        // (B) 先算内部行 (2 .. ny‑3)，可与网路传输并行
        // 这个do_accel参数很重要，防止重复加速，对于最后一个进程，只需要在这里加速一次即可，在后续依赖halo的部分不需要加速
        float tot_u_in  = fusion_more(params, cells, tmp_cells, obstacles,
                                      w11, w22, c_sq, w0, w1, w2,
                                      divideVal, divideVal2,
                                      ThisRank, ProcessSize,
                /*rowStart*/2, /*rowEnd*/params.ny-3, 1);

        // (C) 等待 halo 完成
        // MPI_Waitall(4, req, MPI_STATUSES_IGNORE); 的作用就是 阻塞当前线程，
        // 直到 req[0]‥req[3] 所代表的 4个非阻塞通信全部完成。
//        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        // (D) 再算依赖 halo 的边界行 1 与 ny‑2
        float tot_u_bd  = fusion_more(params, cells, tmp_cells, obstacles,
                                      w11, w22, c_sq, w0, w1, w2,
                                      divideVal, divideVal2,
                                      ThisRank, ProcessSize, 1, 1, 0);
        tot_u_bd +=      fusion_more(params, cells, tmp_cells, obstacles,
                                     w11, w22, c_sq, w0, w1, w2,
                                     divideVal, divideVal2,
                                     ThisRank, ProcessSize, params.ny-2, params.ny-2, 0);

        // (E) 汇总本轮速度并交换指针
        av_vels[tt] = tot_u_in + tot_u_bd;

        t_speed* temp = cells;   // pointer swap
        cells          = tmp_cells;
        tmp_cells      = temp;
    }

    /* Compute time stops here, collate time starts*/
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic=comp_toc;

    // Collate data from ranks here

    // 下面的结果如果和上面不一样，说明拷贝出了问题！

    // 主进程在这回收数据, 回收的时候要注意，每个cells只有一部分是真正的数据，还有两行是halo
    // 回收cells，并且放到results中
    // obstacles我觉得不用回收，主进程保留一个大obstacles
    // main process collect data here, and the operation is blocking operation

    // 这一部分可以用MPI_Gatherv来实现，gatherv允许每个进程发送不同数量的数据，这样就不用计算每个进程需要发送的数据量了
    if (ThisRank==0) {
        // 接收每个进程的cells数据
        // 保存在results中
        // 先把自己的数据拷贝到results中,results的大小就是正常大板子的大小！
        for (int i = 1; i < params.ny - 1; i++) {
            for (int j = 0; j < params.nx; j++) {
                results[j + (i-1) * params.nx] = cells[j + i * params.nx];
            }
        }
        int basic_work = (KeepTotalRows - 3) / ProcessSize; // 每个进程至少处理的行数
        int remainder = (KeepTotalRows - 3) % ProcessSize; // 余数行

        // offsetStart初始化为主进程要处理的大小
        int offsetStart = local_n_Size;

        // 一共112个进程
        for (int i = 1; i < ProcessSize; i++) {
            // 重新计算每个进程需要处理的行数以及元素总数
            // 为每个进程计算行数
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

    //这里是为了计算每轮的平均值
    float total_av_vels[params.maxIters];
    // 这行代码的作用是让所有进程把自己 av_vels 数组里的浮点值按元素位置做「求和」运算，
    // 并把结果汇总到秩为0 的进程上的 total_av_vels 数组中。具体含义：
    // total_avels：只有当进程秩（rank）为 0 时才写入规约结果，其他进程对此参数可忽略
    MPI_Reduce(av_vels, total_av_vels, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ThisRank == 0){
        // 计算每轮的平均值
        for (int tt = 0; tt < params.maxIters; tt++){
            av_vels[tt] = total_av_vels[tt] / (float)numberOfNonObstacles;
        }
    }

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;
//    /* write final values and free memory */
// 在主进程写入数据，然后make check检查应该就是检查这里写的数据。
    if (ThisRank == 0){
        // 在写之前把正确的params->ny赋值回去
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
    // 这里只对最后一个进程进行加速操作
    // 如果是最后一个进程，那么对倒数第二行进行加速操作
    if (do_accel==1 && ThisRank == ProcessSize-1 ){
        const int LastSecondRow = params.ny - 3;
        // LastSecondRow = 3
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* if the cell is not occupied and
            ** we don't send a negative density */

            // 这里注意一下，障碍物索引
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
    
    //all above is accelerate_flow()!!!!!
    float tot_u;          /* accumulated magnitudes of velocity for each cell */
    /* initialise */
    tot_u = 0.f;

    /* loop over _all_ cells */
    // jj index from 1 to params.ny-2
    // ii index from 0 to params.nx-1
    // 因为jj=0和最后一行是halo部分，是其他进程的halo数据，所以此进程要处理的数据从1开始到params.ny-2
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

                // 注意如果是rebound，他的speeds[0]是不变的
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

    //主进程也处理任务

    // 我需要在这里让最后一个进程，最少处理3行，这三行是最后三行！！！！！
    // 这个原因不用管，是因为加速的逻辑，加速是只加速整个板子的倒数第二行，所以最后一个进程要处理最后三行防止数据冲突

//    目的：保证最后一个进程至少拥有网格倒数第 3、2、1 行，以便执行 accelerate_flow（只对 全局倒数第二行 加速）。
//
//    局部网格尺寸：local_n = rows_per_process * nx
//
//    真正参与计算的数据行：1 … rows_per_process，第 0 行和最后一行是 halo。

    int basic_work =  (params->ny-3) / ProcessSize; // 每个进程至少处理的行数
    int remainder = (params->ny-3) % ProcessSize; // 余数行
    // 假如size=8，余数最多是7行当rows=15时，不建议让最后一个线程处理8行，可以把这7行平均分给前7个线程
    // 这样前面7个进程每个处理2行，最后一个进程处理1行

    // 为每个进程计算行数
    int rows_per_process = basic_work + (ThisRank < remainder ? 1 : 0);
    // 最后一个进程多处理3行
    if (ThisRank == ProcessSize-1){
        rows_per_process += 3;
    }

    // 为每个进程分配本地数组大小,一个进程要处理的元素个数
    // 最后一个进程的local_n可能会和其他进程不一样
    int local_n = rows_per_process * params->nx;

    // 为每个进程分配本地数组大小，并且多分配两行用来存储halo数据，
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (local_n + 2*params->nx));

    // 同样为tmp_cells分配本地数组大小
    // 这个不用初始化
    *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (local_n + 2*params->nx));

    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    // 初始值都是一样的，无所谓。
    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density      / 9.f;
    float w2 = params->density      / 36.f;

    // 初始化这个进程的值，每个进程初始化的值一样，无所谓
    // 注意，可以也初始化一下上下边界，初始化成一样的值，无所谓
    // 因为不初始化，他halo exchange也会把这两行初始化
    for (int jj = 1; jj < rows_per_process+1; jj++)
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
        // 这个是用来放所有进程的结果的，主进程用来收集所有进程的数据
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
        // 一共有多少个物体 params->ny*params->nx  16384
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

        // 此时，主进程已经读取了障碍物，接下来广播（实际是发送）给其他进程，每个其他进程收到的obstacles对应各自的实际要处理的cells的大小
        // 主进程也分配自己的obstacles
        // 这个local_n是主进程自己的local_n
        *obstacles_ptr = malloc(sizeof(int) * local_n);
        for (int i = 0; i < local_n; i++) {
            (*obstacles_ptr)[i] = (*obstacles_ptr_Total)[i];
        }
        // 把其他obstacles发送给其他进程，每个进程的local_n不一样，因为这是在主进程中，所以要重新计算大小
        for (int i = 1; i < ProcessSize; i++) {
            int rows_for_this_process = basic_work + (i < remainder ? 1 : 0);
            if (i == ProcessSize-1){
                rows_for_this_process += 3;
            }

            // 为每个进程分配本地数组大小,一个进程要处理的元素个数
            int size_Process = rows_for_this_process * params->nx;

            int start_row = 0;
            // 把之前每个进程处理的行数都加起来，得到这个进程开始的行数
            for (int t = 0; t < i; t++) {
                int rows_for_previous_process = basic_work + (t < remainder ? 1 : 0);
                start_row += rows_for_previous_process;
            }

            // 为每个进程分配本地数组大小，不用分配halo数据
            // 这里分配一个临时变量，用来发送数据
            // 其实这一部分可以省略，可以直接发送obstacles_ptr_Total，计算一下索引即可
            int sendArr[size_Process];
            for (int j = 0; j < size_Process; j++) {
                sendArr[j] = (*obstacles_ptr_Total)[j + (start_row * params->nx)];
            }
            // 这里的size_Process是每个进程的local_n
            MPI_Send(sendArr, size_Process, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }else{
        // 这里是其他进程
        // 其他进程接收主进程广播的obstacles
        *obstacles_ptr = malloc(sizeof(int) * local_n);
        // printf the size of local_n for this process
        MPI_Recv(*obstacles_ptr, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // here to make sure nx and ny are small fractions

    params->nx = params->nx;
    // 真正要处理的数据是从第1行到rows_per_process+1行
    // the real handle data is from the first row to the rows_per_process+1 row

    // 在这确保params的ny是片段的大小，方便后续计算
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
