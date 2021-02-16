#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL EUPHONIC_NPY_ARRAY_API
#include <string.h>
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "load_libs.h"
#include "dyn_mat.h"
#include "py_util.h"
#include "util.h"

// Headers for profiling
#include <stdbool.h>
#include <time.h>

static PyObject *calculate_phonons(PyObject *self, PyObject *args) {

    // Define input args
    PyObject *py_idata; // ForceConstants instance
    PyArrayObject *py_cell_vec;
    PyArrayObject *py_recip_vec;
    PyArrayObject *py_rqpts;
    PyArrayObject *py_split_idx;
    PyArrayObject *py_q_dirs;
    PyArrayObject *py_fc;
    PyArrayObject *py_sc_ogs;
    PyArrayObject *py_asr_correction;
    PyArrayObject *py_dmat_weighting;
    PyArrayObject *py_evals;
    PyArrayObject *py_dmats;
    int dipole;
    int reciprocal_asr;
    int splitting;
    int n_threads = 1;
    const char *scipy_dir;

    // Define vars to be obtained from ForceConstants attributes
    PyObject *py_crystal; // Crystal object
    PyArrayObject *py_n_sc_ims;
    PyArrayObject *py_sc_im_idx;
    PyArrayObject *py_cell_ogs;
    // Extra vars only required if dipole = True
    PyArrayObject *py_born;
    PyArrayObject *py_dielectric;
    double eta;
    PyArrayObject *py_H_ab;
    PyArrayObject *py_dipole_cells;
    PyArrayObject *py_gvec_phases;
    PyArrayObject *py_gvecs_cart;
    PyArrayObject *py_dipole_q0;

    // Vars to be obtained from Crystal attributes
    int n_atoms;
    PyArrayObject *py_atom_r;

    // Define pointers to Python array data
    double *cell_vec;
    double *recip_vec;
    double *rqpts;
    int *split_idx;
    double *q_dirs;
    double *fc;
    int *sc_ogs;
    double *asr_correction;
    double *dmat_weighting;
    double *evals;
    double *dmats;
    int *n_sc_ims;
    int *sc_im_idx;
    int *cell_ogs;
    // Extra vars only required if dipole = True
    double *atom_r;
    double *born;
    double *dielectric;
    double *H_ab;
    double *dipole_cells;
    double *gvec_phases;
    double *gvecs_cart;
    double *dipole_q0;

    // Other vars
    int n_cells;
    int n_rqpts;
    int dmats_len;
    int n_split_qpts;
    int q, i, qpos;
    int max_ims;
    int dmat_elems;
    // Extra vars only required if dipole = True
    int n_dipole_cells;
    int n_gvecs;

    // Parse inputs
    if (!PyArg_ParseTuple(args, "OO!O!O!O!O!O!O!O!O!iiiO!O!is",
                          &py_idata,
                          &PyArray_Type, &py_cell_vec,
                          &PyArray_Type, &py_recip_vec,
                          &PyArray_Type, &py_rqpts,
                          &PyArray_Type, &py_split_idx,
                          &PyArray_Type, &py_q_dirs,
                          &PyArray_Type, &py_fc,
                          &PyArray_Type, &py_sc_ogs,
                          &PyArray_Type, &py_asr_correction,
                          &PyArray_Type, &py_dmat_weighting,
                          &dipole,
                          &reciprocal_asr,
                          &splitting,
                          &PyArray_Type, &py_evals,
                          &PyArray_Type, &py_dmats,
                          &n_threads,
                          &scipy_dir)) {
        return NULL;
    }

    // Get rest of vars from ForceConstants object
    if (attr_from_pyobj(py_idata, "crystal", &py_crystal) ||
        attr_from_pyobj(py_idata, "_n_sc_images", &py_n_sc_ims) ||
        attr_from_pyobj(py_idata, "_sc_image_i", &py_sc_im_idx) ||
        attr_from_pyobj(py_idata, "cell_origins", &py_cell_ogs)) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to read attributes from object\n");
            return NULL;
    }
    if (dipole) {
        if (attr_from_pyobj(py_idata, "_born", &py_born) ||
            attr_from_pyobj(py_idata, "_dielectric", &py_dielectric) ||
            double_from_pyobj(py_idata, "_eta", &eta) ||
            attr_from_pyobj(py_idata, "_H_ab", &py_H_ab) ||
            attr_from_pyobj(py_idata, "_cells", &py_dipole_cells) ||
            attr_from_pyobj(py_idata, "_gvec_phases", &py_gvec_phases) ||
            attr_from_pyobj(py_idata, "_gvecs_cart", &py_gvecs_cart) ||
            attr_from_pyobj(py_idata, "_dipole_q0", &py_dipole_q0)) {
                PyErr_Format(PyExc_RuntimeError,
                             "Failed to read dipole attributes from object\n");
                return NULL;
        }
    }
    // Get vars from Crystal object
    if (int_from_pyobj(py_crystal, "n_atoms", &n_atoms) ||
        attr_from_pyobj(py_crystal, "atom_r", &py_atom_r)) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to read attributes from Crystal object\n");
            return NULL;
    }

    // Point to Python array data
    cell_vec = (double*) PyArray_DATA(py_cell_vec);
    recip_vec = (double*) PyArray_DATA(py_recip_vec);
    rqpts = (double*) PyArray_DATA(py_rqpts);
    split_idx = (int*) PyArray_DATA(py_split_idx);
    q_dirs = (double*) PyArray_DATA(py_q_dirs);
    fc = (double*) PyArray_DATA(py_fc);
    sc_ogs = (int*) PyArray_DATA(py_sc_ogs);
    asr_correction = (double*) PyArray_DATA(py_asr_correction);
    dmat_weighting = (double*) PyArray_DATA(py_dmat_weighting);
    evals = (double*) PyArray_DATA(py_evals);
    dmats = (double*) PyArray_DATA(py_dmats);
    n_sc_ims = (int*) PyArray_DATA(py_n_sc_ims);
    sc_im_idx = (int*) PyArray_DATA(py_sc_im_idx);
    cell_ogs = (int*) PyArray_DATA(py_cell_ogs);
    n_cells = PyArray_DIMS(py_fc)[0];
    n_rqpts = PyArray_DIMS(py_rqpts)[0];
    n_split_qpts = PyArray_DIMS(py_split_idx)[0];
    dmats_len = PyArray_DIMS(py_dmats)[0];
    max_ims = PyArray_DIMS(py_sc_im_idx)[3];
    dmat_elems = 2*9*n_atoms*n_atoms;
    if (dipole) {
        atom_r = (double*) PyArray_DATA(py_atom_r);
        born = (double*) PyArray_DATA(py_born);
        dielectric = (double*) PyArray_DATA(py_dielectric);
        H_ab = (double*) PyArray_DATA(py_H_ab);
        dipole_cells = (double*) PyArray_DATA(py_dipole_cells);
        gvec_phases = (double*) PyArray_DATA(py_gvec_phases);
        gvecs_cart = (double*) PyArray_DATA(py_gvecs_cart);
        dipole_q0 = (double*) PyArray_DATA(py_dipole_q0);
        n_dipole_cells = PyArray_DIMS(py_dipole_cells)[0];
        n_gvecs = PyArray_DIMS(py_gvec_phases)[0];
    }

    // Load library functions
    ZheevdFunc zheevd;
    zheevd = get_zheevd(scipy_dir);
    if (zheevd == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Could not load zheevd function\n");
        return NULL;
    }

    bool profile = true;

    omp_set_num_threads(n_threads);
    struct timespec totpar_ti, totpar_tf;
    double total_par_time;
    if (profile) {
        clock_gettime(CLOCK_REALTIME, &totpar_ti);
    }
    #pragma omp parallel
    {
        int *n_calls;
        double *call_time;
        struct timespec ti, tf;
        struct timespec totfor_ti, totfor_tf;
        double total_for_time;
        if (profile) {
            total_for_time = 0;
            n_calls = calloc(5, sizeof(int));
            call_time = calloc(5, sizeof(double));
        }

        double *corr, *dmat_per_q;
        if (dipole) {
            corr = (double*) malloc(dmat_elems*sizeof(double));
        }
        // If space for the eigenvectors has not been allocated, assume they
        // aren't to be returned and just allocate just enough memory for each
        // q-point calculation
        if (dmats_len == 0) {
            dmat_per_q = (double*) malloc(dmat_elems*sizeof(double));
        }

        #pragma omp for
        for (q = 0; q < n_rqpts; q++) {
            if (profile) clock_gettime(CLOCK_REALTIME, &totfor_ti);

            double *qpt, *dmat, *eval;
            qpt = (rqpts + 3*q);
            eval = (evals + q*3*n_atoms);

            if (dmats_len == 0) {
                dmat = dmat_per_q;
            } else {
                dmat = (dmats + q*dmat_elems);
            }

            if (profile) clock_gettime(CLOCK_REALTIME, &ti);
            calculate_dyn_mat_at_q(qpt, n_atoms, n_cells, max_ims, n_sc_ims,
                sc_im_idx, cell_ogs, sc_ogs, fc, dmat);
            if (profile) {
                clock_gettime(CLOCK_REALTIME, &tf);
                n_calls[0]++;
                call_time[0] = call_time[0] + (tf.tv_sec - ti.tv_sec)
                    + (tf.tv_nsec - ti.tv_nsec)/((double) 1e9);
            }

            if (dipole) {
                if (profile) clock_gettime(CLOCK_REALTIME, &ti);
                calculate_dipole_correction(qpt, n_atoms, cell_vec, recip_vec,
                    atom_r, born, dielectric, H_ab, dipole_cells,
                    n_dipole_cells, gvec_phases, gvecs_cart, n_gvecs,
                    dipole_q0, eta, corr);
                if (profile) {
                    clock_gettime(CLOCK_REALTIME, &tf);
                    n_calls[1]++;
                    call_time[1] = call_time[1] + (tf.tv_sec - ti.tv_sec)
                        + (tf.tv_nsec - ti.tv_nsec)/((double) 1e9);
                }
                add_arrays(dmat_elems, corr, dmat);
            }

            if (reciprocal_asr) {
                add_arrays(dmat_elems, asr_correction, dmat);
            }

            // Calculate non-analytical correction for LO-TO splitting
            if (splitting && is_gamma(qpt)) {
                // Find q-direction for this q-point
                qpos = -1;
                for (i = 0; i < n_split_qpts; i++) {
                    if (split_idx[i] == q) {
                        qpos = i;
                        break;
                    }
                }
                calculate_gamma_correction((q_dirs + 3*qpos), n_atoms,
                    cell_vec, recip_vec, born, dielectric, corr);
                add_arrays(dmat_elems, corr, dmat);
            }

            mass_weight_dyn_mat(dmat_weighting, n_atoms, dmat);
            if (profile) clock_gettime(CLOCK_REALTIME, &ti);
            diagonalise_dyn_mat_zheevd(n_atoms, qpt, dmat, eval, zheevd);
            if (profile) {
                clock_gettime(CLOCK_REALTIME, &tf);
                n_calls[2]++;
                call_time[2] = call_time[2] + (tf.tv_sec - ti.tv_sec)
                    + (tf.tv_nsec - ti.tv_nsec)/((double) 1e9);
            }
            evals_to_freqs(n_atoms, eval);

            if (profile) {
                clock_gettime(CLOCK_REALTIME, &totfor_tf);
                total_for_time = total_for_time + (totfor_tf.tv_sec - totfor_ti.tv_sec)
                    + (totfor_tf.tv_nsec - totfor_ti.tv_nsec)/((double) 1e9);
            }

        }
        if (profile) {
            char filename[sizeof "euphonic_c_ext.99.profile"];
            FILE *fptr;
            sprintf(filename, "euphonic_c_ext.%02d.profile",
                    omp_get_thread_num());
            fptr = fopen(filename, "w");
            fprintf(fptr, "%-30s %06d %8.4f\n",
                    "total in for loop", 1, total_for_time);
            fprintf(fptr, "%-30s %06d %8.4f\n",
                    "calculate_dyn_mat_at_q", n_calls[0], call_time[0]);
            fprintf(fptr, "%-30s %06d %8.4f\n",
                    "calculate_dipole_correction", n_calls[1], call_time[1]);
            fprintf(fptr, "%-30s %06d %8.4f\n",
                    "diagonalise_dyn_mat", n_calls[2], call_time[2]);
            fclose(fptr);
        }
    }
    if (profile) {
        clock_gettime(CLOCK_REALTIME, &totpar_tf);
        total_par_time = (totpar_tf.tv_sec - totpar_ti.tv_sec)
            + (totpar_tf.tv_nsec - totpar_ti.tv_nsec)/((double) 1e9);
        FILE *fptr;
        fptr = fopen("euphonic_c_ext.00.profile", "a");
        fprintf(fptr, "%-30s %06d %8.4f\n",
                "total in parallel section", 1, total_par_time);
        fclose(fptr);
    }
    return Py_None;
}

static PyMethodDef _euphonic_methods[] = {
    {"calculate_phonons", calculate_phonons, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef _euphonic_module_def = {
    PyModuleDef_HEAD_INIT,
    "_euphonic",
    NULL,
    -1,
    _euphonic_methods
};

PyMODINIT_FUNC PyInit__euphonic(void) {
    import_array();
    return PyModule_Create(&_euphonic_module_def);
}
#else
PyMODINIT_FUNC init_euphonic() {
    import_array();
    Py_InitModule3("_euphonic", _euphonic_methods, NULL);
}
#endif
