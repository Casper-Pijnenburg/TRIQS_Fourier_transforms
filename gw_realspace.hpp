namespace triqs_tprf {
    b_g_Dw_t polarization(b_g_Dw_cvt g_w, dlr_imfreq iw_mesh_b, int num_cores = 0);
    b_g_Dw_t screened_potential(b_g_Dw_cvt P, matrix<double> V, bool self_interactions, int num_cores);
    b_g_Dw_t dyn_self_energy(b_g_Dw_cvt G, b_g_Dw_cvt W, matrix<double> V, bool self_interactions, int num_cores);
    b_g_Dw_t hartree_self_energy(b_g_Dw_cvt G, matrix<double> V, bool self_interactions, int num_cores);
    b_g_Dw_t fock_self_energy(b_g_Dw_cvt G, matrix<double> V, bool self_interactions, int num_cores);
    b_g_Dt_t iw_to_tau_p(b_g_Dw_cvt g_w, int num_cores);
    b_g_Dw_t tau_to_iw_p(b_g_Dt_cvt g_t, int num_cores);
    b_g_Dt_t iw_to_tau(b_g_Dw_cvt g_w);
    b_g_Dw_t tau_to_iw(b_g_Dt_cvt g_w);
}

