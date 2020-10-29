import inspect

import numpy as np

from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic import ureg, Quantity, Crystal


class DebyeWaller(object):
    """
    Stores the (n_ions, 3, 3) anisotropic Debye-Waller exponent
    W_ab to be used in structure factor calculations

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    debye_waller : (n_ions, 3, 3) float Quantity
        The anisotropic Debye-Waller exponent W_ab for each atom, where
        the Debye-Waller factor is exp(-W_ab*Q_a*Q_b) where a,b run over
        the 3 Cartesian directions
    temperature : float Quantity
        The temperature the Debye-Waller exponent was calculated at
    """

    def __init__(self, crystal, debye_waller, temperature):
        """
        Parameters
        ----------
        crystal : Crystal
            Lattice and atom information
        debye_waller : (n_atoms, 3, 3) Quantity
            The anisotropic Debye-Waller exponent W_ab for each atom,
            where the Debye-Waller factor is exp(-W_ab*Q_a*Q_b) where
            a,b run over the 3 Cartesian directions
        temperature : float Quantity
            The temperature the Debye-Waller exponent was calculated at
        """
        _check_constructor_inputs([crystal], [Crystal], [()], ['crystal'])
        n_atoms = crystal.n_atoms
        _check_constructor_inputs(
            [debye_waller, temperature],
            [Quantity, Quantity],
            [(n_atoms, 3, 3), ()],
            inspect.getfullargspec(self.__init__)[0][2:])
        self.crystal = crystal
        self._debye_waller = debye_waller.to(ureg.bohr**2).magnitude
        self._temperature = temperature.to(ureg.K).magnitude

        self.debye_waller_unit = str(debye_waller.units)
        self.temperature_unit = str(temperature.units)

    @property
    def debye_waller(self):
        return self._debye_waller*ureg('bohr**2').to(
            self.debye_waller_unit)

    @property
    def temperature(self):
        # See https://pint.readthedocs.io/en/latest/nonmult.html
        return Quantity(self._temperature, ureg('K')).to(self.temperature_unit)

    def __setattr__(self, name, value):
        _check_unit_conversion(self, name, value,
                               ['debye_waller_unit', 'temperature_unit'])
        super(DebyeWaller, self).__setattr__(name, value)

    def to_dict(self):
        """
        Convert to a dictionary. See DebyeWaller.from_dict for details
        on keys/values

        Returns
        -------
        dict
        """
        dout = _obj_to_dict(self, ['crystal', 'debye_waller', 'temperature'])
        return dout

    def to_json_file(self, filename):
        """
        Write to a JSON file. JSON fields are equivalent to
        DebyeWaller.from_dict keys

        Parameters
        ----------
        filename : str
            Name of the JSON file to write to
        """
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary to a DebyeWaller object

        Parameters
        ----------
        d : dict
            A dictionary with the following keys/values:

            - 'crystal': dict, see Crystal.from_dict
            - 'debye_waller': (n_atoms, 3, 3) float ndarray
            - 'debye_waller_unit': str
            - 'temperature': float
            - 'temperature_unit': str

        Returns
        -------
        DebyeWaller
        """
        crystal = Crystal.from_dict(d['crystal'])
        d = _process_dict(d, quantities=['debye_waller', 'temperature'],
                          optional=['temperature'])
        return DebyeWaller(crystal, d['debye_waller'], d['temperature'])

    @classmethod
    def from_json_file(cls, filename):
        """
        Read from a JSON file. See DebyeWaller.from_dict for required
        fields

        Parameters
        ----------
        filename : str
            The file to read from

        Returns
        -------
        DebyeWaller
        """
        return _obj_from_json_file(cls, filename)


def _calculate_debye_waller(qpts, frequencies, eigenvectors, atom_masses,
                            temperature, weights=None):
    kB = (1*ureg.k).to('hartree/K').magnitude
    mass_term = 1/(4*atom_masses)
    n_qpts = len(qpts)
    if weights is None:
        weights = np.full(n_qpts, 1/n_qpts)

    # Determine q-points near the gamma point and mask out their
    # acoustic modes due to the potentially large 1/frequency factor
    TOL = 1e-8
    is_small_q = np.sum(np.square(qpts), axis=1) < TOL
    freq_mask = np.ones(frequencies.shape, dtype=np.int32)
    freq_mask[is_small_q, :3] = 0

    if temperature > 0:
        x = frequencies/(2*kB*temperature)
        freq_term = 1/(frequencies*np.tanh(x))
    else:
        freq_term = 1/frequencies
    dw = np.zeros((len(atom_masses), 3, 3))

    # Calculating the e.e* term is expensive, do in chunks
    chunk = 1000
    for i in range(int((n_qpts - 1)/chunk) + 1):
        qi = i*chunk
        qf = min((i + 1)*chunk, n_qpts)

        evec_term = np.real(
            np.einsum('ijkl,ijkm->ijklm',
                      eigenvectors[qi:qf],
                      np.conj(eigenvectors[qi:qf])))

        dw += (np.einsum('i,k,ij,ij,ijklm->klm',
                         weights[qi:qf], mass_term, freq_term[qi:qf],
                         freq_mask[qi:qf], evec_term))
    dw = dw/np.sum(weights)

    return dw
