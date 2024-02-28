In our use-case of writing a real-space GW-solver, we're using Block Green's functions defined on imaginary DLR meshes with target shapes of `orbitals` by `orbitals`.

The Fourier transforms we have to perform on the Green's functions seem to take on the bulk of the computation time. Transforming into a coefficient representation with `make_gf_dlr` and then transforming using either `make_gf_dlr_imtime` or `make_gf_dlr_imfreq` is not parallel native within TRQIS so here we can gain a speed-up.

I followed Hugo's parallel implementation for Green's functions on double meshes within TPRF. Instead of transforming the entire Block Green's function at one on a single core, I store rows of the data into smaller Green's functions of size `{1, orbitals}` and transform these seperately for each row in each block for the Block Green's function, running this in parallel and storing the data into a final transformed Block Green's function.

I wrote a simple Python script that imports these parallel functions that are written in C++ within TPRF. A curious thing I noticed is that restricting my parallel implementation to a single core is still quicker than transforming the entire Green's function in one go. I use `np.testing.assert_allclose()` to make sure we get the same result and I've plotted the execution times as a function of the number of cores utilized (just for my 8 core machine).

Here we can see a clear befenit in execution time (200 orbitals, so two 200 by 200 Gfs), but it doesn't seem to scale that well with an increasing number of cores. Results shown in execution_times.pdf.

