import numpy as np
from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq, MeshDLRImTime
import numpy as np
import matplotlib.pylab as plt
import time
import multiprocessing

from triqs_tprf.lattice import iw_to_tau
from triqs_tprf.lattice import iw_to_tau_p
from triqs_tprf.lattice import tau_to_iw
from triqs_tprf.lattice import tau_to_iw_p

def get_cpu_info():
    cpu_info = {}
    with open('/proc/cpuinfo', 'r') as cpuinfo:
        for line in cpuinfo:
            if line.strip():
                key, value = line.strip().split(':')
                cpu_info[key.strip()] = value.strip()
    return cpu_info

def get_threads_per_core():
    cpu_info = get_cpu_info()
    siblings = int(cpu_info.get('siblings', 1))
    cores = int(cpu_info.get('cpu cores', 1))
    threads_per_core = siblings // cores
    return threads_per_core


def generate_G(tij, mesh, spin_names = ['up', 'dn']):
    g_inv = Gf(mesh = mesh, target_shape = np.shape(tij))
    g_inv << iOmega_n - tij.transpose()
    g = g_inv.inverse()
    return BlockGf(block_list = [g] * 2, name_list = spin_names, make_copies = False)


def main():
    orbitals = 500
    t = 1.0
    tij = np.zeros([orbitals] * 2)
    for i in range(orbitals - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t


    iw_mesh = MeshDLRImFreq(beta = 100, statistic = 'Fermion', w_max = 20.0, eps = 1e-12, symmetrize = True)
    G_iw = generate_G(tij, iw_mesh)

    start = time.perf_counter()
    G_tau_single_core = iw_to_tau(G_iw)
    execution_time_single_core = time.perf_counter() - start



    available_cores = multiprocessing.cpu_count() // get_threads_per_core()
    execution_times = []
    for num_cores in range(1, available_cores + 1):
        start = time.perf_counter()
        G_tau_parallel = iw_to_tau_p(G_iw, num_cores)
        execution_times.append(time.perf_counter() - start)

        np.testing.assert_allclose(G_tau_single_core['up'].data, G_tau_parallel['up'].data)
        np.testing.assert_allclose(G_tau_single_core['dn'].data, G_tau_parallel['dn'].data)

    fig, ax = plt.subplots(figsize = (6, 6), tight_layout = True)
    ax.plot(range(1, available_cores + 1), execution_times, marker = 'o', color = 'red', label = 'Parallel implementation')
    ax.hlines(execution_time_single_core, 0.5, available_cores + 0.5, color = 'black', label = 'TRIQS implementation')
    ax.set_xlim(0.5, available_cores + 0.5)
    ax.set_ylim(0, 1.1 * np.max([execution_time_single_core, np.max(execution_times)]))
    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Execution time (s)')
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xticks(range(1, available_cores + 1))


    # ax.text(0.2, 0.85, f"Number of orbitals indices = {orbitals}", horizontalalignment = 'left', verticalalignment = 'center', transform = ax.transAxes, fontsize = 15)
    # ax.text(0.2, 0.8, f"Number of DLR mesh points = {len(list(mesh.values()))}", horizontalalignment = 'left', verticalalignment = 'center', transform = ax.transAxes, fontsize = 15)
    plt.savefig("execution_times.pdf")

    return

if __name__ == '__main__':
    main()