/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_crane_mpc_pzs100_H_
#define ACADOS_SOLVER_crane_mpc_pzs100_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define CRANE_MPC_PZS100_NX     23
#define CRANE_MPC_PZS100_NZ     0
#define CRANE_MPC_PZS100_NU     5
#define CRANE_MPC_PZS100_NP     11
#define CRANE_MPC_PZS100_NP_GLOBAL     0
#define CRANE_MPC_PZS100_NBX    18
#define CRANE_MPC_PZS100_NBX0   23
#define CRANE_MPC_PZS100_NBU    5
#define CRANE_MPC_PZS100_NSBX   4
#define CRANE_MPC_PZS100_NSBU   0
#define CRANE_MPC_PZS100_NSH    6
#define CRANE_MPC_PZS100_NSH0   6
#define CRANE_MPC_PZS100_NSG    0
#define CRANE_MPC_PZS100_NSPHI  0
#define CRANE_MPC_PZS100_NSHN   0
#define CRANE_MPC_PZS100_NSGN   0
#define CRANE_MPC_PZS100_NSPHIN 0
#define CRANE_MPC_PZS100_NSPHI0 0
#define CRANE_MPC_PZS100_NSBXN  4
#define CRANE_MPC_PZS100_NS     10
#define CRANE_MPC_PZS100_NS0    6
#define CRANE_MPC_PZS100_NSN    4
#define CRANE_MPC_PZS100_NG     0
#define CRANE_MPC_PZS100_NBXN   18
#define CRANE_MPC_PZS100_NGN    0
#define CRANE_MPC_PZS100_NY0    24
#define CRANE_MPC_PZS100_NY     24
#define CRANE_MPC_PZS100_NYN    14
#define CRANE_MPC_PZS100_N      49
#define CRANE_MPC_PZS100_NH     6
#define CRANE_MPC_PZS100_NHN    0
#define CRANE_MPC_PZS100_NH0    6
#define CRANE_MPC_PZS100_NPHI0  0
#define CRANE_MPC_PZS100_NPHI   0
#define CRANE_MPC_PZS100_NPHIN  0
#define CRANE_MPC_PZS100_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct crane_mpc_pzs100_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *impl_dae_fun;
    external_function_external_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_external_param_casadi *impl_dae_jac_x_xdot_u_z;




    // cost

    external_function_external_param_casadi *cost_y_fun;
    external_function_external_param_casadi *cost_y_fun_jac_ut_xt;



    external_function_external_param_casadi cost_y_0_fun;
    external_function_external_param_casadi cost_y_0_fun_jac_ut_xt;



    external_function_external_param_casadi cost_y_e_fun;
    external_function_external_param_casadi cost_y_e_fun_jac_ut_xt;


    // constraints
    external_function_external_param_casadi *nl_constr_h_fun_jac;
    external_function_external_param_casadi *nl_constr_h_fun;





    external_function_external_param_casadi nl_constr_h_0_fun_jac;
    external_function_external_param_casadi nl_constr_h_0_fun;






} crane_mpc_pzs100_solver_capsule;

ACADOS_SYMBOL_EXPORT crane_mpc_pzs100_solver_capsule * crane_mpc_pzs100_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_free_capsule(crane_mpc_pzs100_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_create(crane_mpc_pzs100_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_reset(crane_mpc_pzs100_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of crane_mpc_pzs100_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_create_with_discretization(crane_mpc_pzs100_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_update_time_steps(crane_mpc_pzs100_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_update_qp_solver_cond_N(crane_mpc_pzs100_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_update_params(crane_mpc_pzs100_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_update_params_sparse(crane_mpc_pzs100_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_set_p_global_and_precompute_dependencies(crane_mpc_pzs100_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_solve(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_setup_qp_matrices_and_factorize(crane_mpc_pzs100_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_free(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void crane_mpc_pzs100_acados_print_stats(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int crane_mpc_pzs100_acados_custom_update(crane_mpc_pzs100_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *crane_mpc_pzs100_acados_get_nlp_in(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *crane_mpc_pzs100_acados_get_nlp_out(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *crane_mpc_pzs100_acados_get_sens_out(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *crane_mpc_pzs100_acados_get_nlp_solver(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *crane_mpc_pzs100_acados_get_nlp_config(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *crane_mpc_pzs100_acados_get_nlp_opts(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *crane_mpc_pzs100_acados_get_nlp_dims(crane_mpc_pzs100_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *crane_mpc_pzs100_acados_get_nlp_plan(crane_mpc_pzs100_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_crane_mpc_pzs100_H_
