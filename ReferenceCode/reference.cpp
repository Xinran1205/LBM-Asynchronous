int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    accelerate_flow(params, cells, obstacles);
    propagate(params, cells, tmp_cells);
    rebound(params, cells, tmp_cells, obstacles);
    collision(params, cells, tmp_cells, obstacles);
    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
    /* compute weighting factors */
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */
    int jj = params.ny - 2;

    for (int ii = 0; ii < params.nx; ii++)
    {
        /* if the cell is not occupied and
        ** we don't send a negative density */
        if (!obstacles[ii + jj*params.nx]
            && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
            && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
            && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
        {
            /* increase 'east-side' densities */
            cells[ii + jj*params.nx].speeds[1] += w1;
            cells[ii + jj*params.nx].speeds[5] += w2;
            cells[ii + jj*params.nx].speeds[8] += w2;
            /* decrease 'west-side' densities */
            cells[ii + jj*params.nx].speeds[3] -= w1;
            cells[ii + jj*params.nx].speeds[6] -= w2;
            cells[ii + jj*params.nx].speeds[7] -= w2;
        }
    }

    return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
    /* loop over _all_ cells */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* determine indices of axis-direction neighbours
            ** respecting periodic boundary conditions (wrap around) */
            int y_n = (jj + 1) % params.ny;
            int x_e = (ii + 1) % params.nx;
            int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
            int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
            tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
            tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
            tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
            tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
            tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
            tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
            tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
            tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
            tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
        }
    }

    return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    /* loop over the cells in the grid */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* if the cell contains an obstacle */
            if (obstacles[jj*params.nx + ii])
            {
                /* called after propagate, so taking values from scratch space
                ** mirroring, and writing into main grid */
                cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
                cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
                cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
                cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
                cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
                cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
                cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
                cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
            }
        }
    }

    return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */

    /* loop over the cells in the grid
    ** NB the collision step is called after
    ** the propagate step and so values of interest
    ** are in the scratch-space grid */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* don't consider occupied cells */
            if (!obstacles[ii + jj*params.nx])
            {
                /* compute local density total */
                float local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
                }

                /* compute x velocity component */
                float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                             + tmp_cells[ii + jj*params.nx].speeds[5]
                             + tmp_cells[ii + jj*params.nx].speeds[8]
                             - (tmp_cells[ii + jj*params.nx].speeds[3]
                                + tmp_cells[ii + jj*params.nx].speeds[6]
                                + tmp_cells[ii + jj*params.nx].speeds[7]))
                            / local_density;
                /* compute y velocity component */
                float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                             + tmp_cells[ii + jj*params.nx].speeds[5]
                             + tmp_cells[ii + jj*params.nx].speeds[6]
                             - (tmp_cells[ii + jj*params.nx].speeds[4]
                                + tmp_cells[ii + jj*params.nx].speeds[7]
                                + tmp_cells[ii + jj*params.nx].speeds[8]))
                            / local_density;

                /* velocity squared */
                float u_sq = u_x * u_x + u_y * u_y;

                /* directional velocity components */
                float u[NSPEEDS];
                u[1] =   u_x;        /* east */
                u[2] =         u_y;  /* north */
                u[3] = - u_x;        /* west */
                u[4] =       - u_y;  /* south */
                u[5] =   u_x + u_y;  /* north-east */
                u[6] = - u_x + u_y;  /* north-west */
                u[7] = - u_x - u_y;  /* south-west */
                u[8] =   u_x - u_y;  /* south-east */

                /* equilibrium densities */
                float d_equ[NSPEEDS];
                /* zero velocity density: weight w0 */
                d_equ[0] = w0 * local_density
                           * (1.f - u_sq / (2.f * c_sq));
                /* axis speeds: weight w1 */
                d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                                 + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));
                d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                                 + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));
                d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                                 + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));
                d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                                 + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));
                /* diagonal speeds: weight w2 */
                d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                                 + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));
                d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                                 + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));
                d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                                 + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));
                d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                                 + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));

                /* relaxation step */
                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                          + params.omega
                                                            * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
                }
            }
        }
    }

    return EXIT_SUCCESS;
}