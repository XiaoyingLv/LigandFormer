# coding:utf-8
"""
Engineered featurizers for ligand based VS project
Created  :  10, 15, 2019
Revised  :  10, 15, 2019
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'

import numpy as np
from rdkit import Chem


################ codes below are from `..ligandbasedpackage.featureloader` ##############################
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):  # [DV] one-hot encoding
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_intervals(l):
    """For list of lists, gets the cumulative products of the lengths"""
    intervals = len(l) * [0]
    # Initalize with 1
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

    return intervals


def safe_index(l, e):
    """Gets the index of e in l, providing an index of len(l) if not found"""
    try:
        return l.index(e)
    except:
        return len(l)


possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']
reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list
]
intervals = get_intervals(reference_lists)


def get_feature_list(atom):
    features = 6 * [0]
    features[0] = safe_index(possible_atom_list, atom.GetSymbol())
    features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
    features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
    features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
    features[4] = safe_index(possible_number_radical_e_list,
                             atom.GetNumRadicalElectrons())
    features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
    return features


def features_to_id(features, intervals):
    """Convert list of features into index using spacings provided in intervals"""
    id = 0
    for k in range(len(intervals)):
        id += features[k] * intervals[k]

    # Allow 0 index to correspond to null molecule 1
    id = id + 1
    return id


def id_to_features(id, intervals):
    features = 6 * [0]

    # Correct for null
    id -= 1

    for k in range(0, 6 - 1):
        # print(6-k-1, id)
        features[6 - k - 1] = id // intervals[6 - k - 1]
        id -= features[6 - k - 1] * intervals[6 - k - 1]
    # Correct for last one
    features[0] = id
    return features


def atom_to_id(atom):
    """Return a unique id corresponding to the atom type"""
    features = get_feature_list(atom)
    return features_to_id(features, intervals)


def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        # [DV] generate feature for "atom"  (why there is "GetDegree()" which should be concept in molecule?)
        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C',
                'N',
                'O',
                'S',
                'F',
                'Si',
                'P',
                'Cl',
                'Br',
                'Mg',
                'Na',
                'Ca',
                'Fe',
                'As',
                'Al',
                'I',
                'B',
                'V',
                'K',
                'Tl',
                'Yb',
                'Sb',
                'Sn',
                'Ag',
                'Pd',
                'Co',
                'Se',
                'Ti',
                'Zn',
                'H',  # H?
                'Li',
                'Ge',
                'Cu',
                'Au',
                'Ni',
                'Cd',
                'In',
                'Mn',
                'Zr',
                'Cr',
                'Pt',
                'Hg',
                'Pb',
                'Unknown'
            ]) + one_of_k_encoding(atom.GetDegree(),
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)


def cumulative_sum_minus_last(l, offset=0):
    """Returns cumulative sums for set of counts, removing last entry.

    Returns the cumulative sums for a set of counts with the first returned value
    starting at 0. I.e [3,2,4] -> [0, 3, 5]. Note last sum element 9 is missing.
    Useful for reindexing

    Parameters
    ----------
    l: list
      List of integers. Typically small counts.
    """
    return np.delete(np.insert(np.cumsum(l), 0, 0), -1) + offset


def cumulative_sum(l, offset=0):
    """Returns cumulative sums for set of counts.

    Returns the cumulative sums for a set of counts with the first returned value
    starting at 0. I.e [3,2,4] -> [0, 3, 5, 9]. Keeps final sum for searching.
    Useful for reindexing.

    Parameters
    ----------
    l: list
      List of integers. Typically small counts.
    """
    return np.insert(np.cumsum(l), 0, 0) + offset


class MultiConvMol(object):
    """Holds information about multiple molecules, for use in feeding information
       into tensorflow. Generated using the agglomerate_mols function
    """

    def __init__(self, nodes, deg_adj_lists, deg_slice, membership, num_mols):
        self.nodes = nodes
        self.deg_adj_lists = deg_adj_lists
        self.deg_slice = deg_slice
        self.membership = membership
        self.num_mols = num_mols
        self.num_atoms = nodes.shape[0]

    def get_deg_adjacency_lists(self):
        return self.deg_adj_lists

    def get_atom_features(self):
        return self.nodes

    def get_num_atoms(self):
        return self.num_atoms

    def get_num_molecules(self):
        return self.num_mols


class ConvMol(object):
    """Holds information about a molecules.

    Resorts order of atoms internally to be in order of increasing degree. Note
    that only heavy atoms (hydrogens excluded) are considered here.
    """

    def __init__(self, atom_features, adj_list, max_deg=10, min_deg=0):
        """
        Parameters
        ----------
        atom_features: np.ndarray
          Has shape (n_atoms, n_feat)
        canon_ad_list: list
          List of length n_atoms, with neighor indices of each atom.
        max_deg: int, optional
          Maximum degree of any atom.
        min_deg: int, optional
          Minimum degree of any atom.
        """

        self.atom_features = atom_features
        self.n_atoms, self.n_feat = atom_features.shape
        self.deg_list = np.array([len(nbrs) for nbrs in adj_list], dtype=np.int32)
        self.canon_adj_list = adj_list
        self.deg_adj_lists = []
        self.deg_slice = []
        self.max_deg = max_deg
        self.min_deg = min_deg

        self.membership = self.get_num_atoms() * [0]

        self._deg_sort()

        # Get the degree id list (which corrects for min_deg)
        self.deg_id_list = np.array(self.deg_list) - min_deg

        # Get the size of each degree block
        deg_size = [
            self.get_num_atoms_with_deg(deg)
            for deg in range(self.min_deg, self.max_deg + 1)
        ]

        # Get the the start indices for items in each block
        self.deg_start = cumulative_sum(deg_size)

        # Get the node indices when they are reset when the degree changes
        deg_block_indices = [
            i - self.deg_start[self.deg_list[i]] for i in range(self.n_atoms)
        ]

        # Convert to numpy array
        self.deg_block_indices = np.array(deg_block_indices)

    def get_atoms_with_deg(self, deg):
        """Retrieves atom_features with the specific degree"""
        start_ind = self.deg_slice[deg - self.min_deg, 0]
        size = self.deg_slice[deg - self.min_deg, 1]
        return self.atom_features[start_ind:(start_ind + size), :]

    def get_num_atoms_with_deg(self, deg):
        """Returns the number of atoms with the given degree"""
        return self.deg_slice[deg - self.min_deg, 1]

    def get_num_atoms(self):
        return self.n_atoms

    def _deg_sort(self):
        """Sorts atoms by degree and reorders internal data structures.

        Sort the order of the atom_features by degree, maintaining original order
        whenever two atom_features have the same degree.
        """
        old_ind = range(self.get_num_atoms())
        deg_list = self.deg_list
        new_ind = list(np.lexsort((old_ind, deg_list)))  # [DV] sort by `deg_list`.

        num_atoms = self.get_num_atoms()

        # Reorder old atom_features
        self.atom_features = self.atom_features[new_ind, :]

        # Reorder old deg lists
        self.deg_list = [self.deg_list[i] for i in new_ind]

        # Sort membership
        self.membership = [self.membership[i] for i in new_ind]  # [DV] all zeros now

        # Create old to new dictionary. not exactly intuitive
        old_to_new = dict(zip(new_ind, old_ind))

        # Reorder adjacency lists
        self.canon_adj_list = [self.canon_adj_list[i] for i in new_ind]
        self.canon_adj_list = [[old_to_new[k]
                                for k in self.canon_adj_list[i]]
                               for i in range(len(new_ind))]

        # Get numpy version of degree list for indexing
        deg_array = np.array(self.deg_list)

        # Initialize adj_lists, which supports min_deg = 1 only   # [DV] todo: what does this mean? `min_deg` can not be other than 1?
        self.deg_adj_lists = (self.max_deg + 1 - self.min_deg) * [0]

        # Parse as deg separated
        for deg in range(self.min_deg, self.max_deg + 1):
            # Get indices corresponding to the current degree
            rng = np.array(range(num_atoms))
            indices = rng[deg_array == deg]

            # Extract and save adjacency list for the current degree
            to_cat = [self.canon_adj_list[i] for i in indices]
            if len(to_cat) > 0:
                adj_list = np.vstack([self.canon_adj_list[i] for i in indices])
                self.deg_adj_lists[deg - self.min_deg] = adj_list

            else:
                self.deg_adj_lists[deg - self.min_deg] = np.zeros(
                    [0, deg], dtype=np.int32)

        # Construct the slice information
        deg_slice = np.zeros([self.max_deg + 1 - self.min_deg, 2], dtype=np.int32)

        for deg in range(self.min_deg, self.max_deg + 1):
            if deg == 0:  # [DV] the if-else here is totally unnecessary
                deg_size = np.sum(deg_array == deg)
            else:
                deg_size = self.deg_adj_lists[deg - self.min_deg].shape[0]

            deg_slice[deg - self.min_deg, 1] = deg_size
            # Get the cumulative indices after the first index
            if deg > self.min_deg:
                deg_slice[deg - self.min_deg, 0] = (
                        deg_slice[deg - self.min_deg - 1, 0] +
                        deg_slice[deg - self.min_deg - 1, 1])

        # Set indices with zero sized slices to zero to avoid indexing errors
        deg_slice[:, 0] *= (deg_slice[:, 1] != 0)
        self.deg_slice = deg_slice

    def get_atom_features(self):
        """Returns canonicalized version of atom features.

        Features are sorted by atom degree, with original order maintained when
        degrees are same.
        """
        return self.atom_features

    def get_adjacency_list(self):
        """Returns a canonicalized adjacency list.

        Canonicalized means that the atoms are re-ordered by degree.

        Returns
        -------
        list
          Canonicalized form of adjacency list.
        """
        return self.canon_adj_list

    def get_deg_adjacency_lists(self):
        """Returns adjacency lists grouped by atom degree.

        Returns
        -------
        list
          Has length (max_deg+1-min_deg). The element at position deg is
          itself a list of the neighbor-lists for atoms with degree deg.
        """
        return self.deg_adj_lists

    def get_deg_slice(self):
        """Returns degree-slice tensor.

        The deg_slice tensor allows indexing into a flattened version of the
        molecule's atoms. Assume atoms are sorted in order of degree. Then
        deg_slice[deg][0] is the starting position for atoms of degree deg in
        flattened list, and deg_slice[deg][1] is the number of atoms with degree deg.

        Note deg_slice has shape (max_deg+1-min_deg, 2).

        Returns
        -------
        deg_slice: np.ndarray
          Shape (max_deg+1-min_deg, 2)
        """
        return self.deg_slice

    # TODO(rbharath): Can this be removed?
    @staticmethod
    def get_null_mol(n_feat, max_deg=10, min_deg=0):
        """Constructs a null molecules

        Get one molecule with one atom of each degree, with all the atoms
        connected to themselves, and containing n_feat features.

        Parameters
        ----------
        n_feat : int
            number of features for the nodes in the null molecule
        """
        # Use random insted of zeros to prevent weird issues with summing to zero
        atom_features = np.random.uniform(0, 1, [max_deg + 1 - min_deg, n_feat])
        canon_adj_list = [
            deg * [deg - min_deg] for deg in range(min_deg, max_deg + 1)
        ]

        return ConvMol(atom_features, canon_adj_list)

    @staticmethod
    def agglomerate_mols(mols, max_deg=10, min_deg=0):
        """Concatenates list of ConvMol's into one mol object that can be used to feed
        into tensorflow placeholders. The indexing of the molecules are preseved during the
        combination, but the indexing of the atoms are greatly changed.

        Parameters
        ----
        mols: list
          ConvMol objects to be combined into one molecule."""

        num_mols = len(mols)

        atoms_per_mol = [mol.get_num_atoms() for mol in mols]  # [DV] list of atom number for each molecule

        # Get atoms by degree
        atoms_by_deg = [
            mol.get_atoms_with_deg(deg)
            for deg in range(min_deg, max_deg + 1)
            for mol in mols
        ]

        # stack the atoms
        all_atoms = np.vstack(atoms_by_deg)

        # Sort all atoms by degree.
        # Get the size of each atom list separated by molecule id, then by degree
        mol_deg_sz = [[mol.get_num_atoms_with_deg(deg)
                       for mol in mols]
                      for deg in range(min_deg, max_deg + 1)]

        # Get the final size of each degree block
        deg_sizes = list(
            map(np.sum, mol_deg_sz))  # (degree_range), here = (11), each element is the number of atoms of given degree
        # Get the index at which each degree starts, not resetting after each degree
        # And not stopping at any speciic molecule

        deg_start = cumulative_sum_minus_last(deg_sizes)
        # Get the tensorflow object required for slicing (deg x 2) matrix, with the
        # first column telling the start indices of each degree block and the
        # second colum telling the size of each degree block

        # Input for tensorflow
        deg_slice = np.array(list(zip(deg_start, deg_sizes)))

        # Determines the membership (atom i belongs to membership[i] molecule)
        membership = [
            k
            for deg in range(min_deg, max_deg + 1) for k in range(num_mols)
            for i in range(mol_deg_sz[deg][k])
        ]

        # Get the index at which each deg starts, resetting after each degree
        # (deg x num_mols) matrix describing the start indices when you count up the atoms
        # in the final representation, stopping at each molecule,
        # resetting every time the degree changes
        start_by_deg = np.vstack([cumulative_sum_minus_last(l) for l in mol_deg_sz])

        # Gets the degree resetting block indices for the atoms in each molecule
        # Here, the indices reset when the molecules change, and reset when the
        # degree changes
        deg_block_indices = [mol.deg_block_indices for mol in mols]

        # Get the degree id lookup list. It allows us to search for the degree of a
        # molecule mol_id with corresponding atom mol_atom_id using
        # deg_id_lists[mol_id,mol_atom_id]
        deg_id_lists = [mol.deg_id_list for mol in mols]

        # This is used for convience in the following function (explained below)
        start_per_mol = deg_start[:, np.newaxis] + start_by_deg

        def to_final_id(mol_atom_id, mol_id):
            # Get the degree id (corrected for min_deg) of the considered atom
            deg_id = deg_id_lists[mol_id][mol_atom_id]

            # Return the final index of atom mol_atom_id in molecule mol_id.  Using
            # the degree of this atom, must find the index in the molecule's original
            # degree block corresponding to degree id deg_id (second term), and then
            # calculate which index this degree block ends up in the final
            # representation (first term). The sum of the two is the final indexn
            return start_per_mol[deg_id,
                                 mol_id] + deg_block_indices[mol_id][mol_atom_id]

        # Initialize the new degree separated adjacency lists
        # [DV] todo: there is no need for `min_deg` param, nor for `+1` length for `deg_adj_lists` below
        deg_adj_lists = [
            np.zeros([deg_sizes[deg], deg], dtype=np.int32)
            for deg in range(min_deg, max_deg + 1)
        ]

        # Update the old adjcency lists with the new atom indices and then combine
        # all together
        for deg in range(min_deg, max_deg + 1):
            row = 0  # Initialize counter
            deg_id = deg - min_deg  # Get corresponding degree id

            # Iterate through all the molecules
            for mol_id in range(num_mols):
                # Get the adjacency lists for this molecule and current degree id
                nbr_list = mols[mol_id].deg_adj_lists[deg_id]

                # Correct all atom indices to the final indices, and then save the
                # results into the new adjacency lists
                for i in range(nbr_list.shape[0]):
                    for j in range(nbr_list.shape[1]):
                        deg_adj_lists[deg_id][row, j] = to_final_id(nbr_list[i, j], mol_id)

                    # Increment once row is done
                    row += 1

        # Get the final aggregated molecule
        concat_mol = MultiConvMol(all_atoms, deg_adj_lists, deg_slice, membership,
                                  num_mols)
        return concat_mol


################ codes above are from `..ligandbasedpackage.featureloader` ##############################

def featurize_A0(mol, master_atom=False, use_chirality=False):
    """
    Extract initial features for molecule, algorithm 0, adapted from `..ligandbasedpackage.featureloader.ConvMolFeaturizer`
    :param mol: `Mol` object from rdkit, represents a molecule
    :param master_atom: Boolean
           if true create a fake atom with bonds to every other atom.
           the initialization is the mean of the other atom features in
           the molecule.  This technique is briefly discussed in
           Neural Message Passing for Quantum Chemistry
           https://arxiv.org/pdf/1704.01212.pdf
    :param use_chirality: flag for chirality
    :return (nodes, neighbours_list), in which `nodes` for vertex feature/embedding, `neighbours_list` for adjacent vertex list
    """
    # Get the node features
    idx_nodes = [(a.GetIdx(), atom_features(a, use_chirality=use_chirality)) for a in mol.GetAtoms()]
    idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
    idx, nodes = list(
        zip(*idx_nodes))  # [DV] now `nodes` contains feature and `idx` contains index of atom within a molecule

    # Stack nodes into an array
    nodes = np.vstack(nodes)  # nodes.shape = (n_atom, n_feature_dim)
    if master_atom:
        master_atom_features = np.expand_dims(np.mean(nodes, axis=0), axis=0)  # [DV] what for? todo: check MPNN
        nodes = np.concatenate([nodes, master_atom_features], axis=0)

    # Get bonds as graph edges, un-directional
    edge_list = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    neighbours_list = [[] for _ in range(len(nodes))]  # list of len() = n_atoms
    for edge in edge_list:
        neighbours_list[edge[0]].append(edge[1])
        neighbours_list[edge[1]].append(edge[0])

    if master_atom:
        fake_atom_index = len(nodes) - 1
        for index in range(len(nodes) - 1):
            neighbours_list[index].append(fake_atom_index)

    return nodes, neighbours_list  # [DV] `nodes`: 2D-array, (n_atom, n_feature), `neighbours_list`: list of list, len=n_atoms
