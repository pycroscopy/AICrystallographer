"""
Utility functions for constructing lattice graphs

Author: Maxim Ziatdinov
"""
import os
import h5py
import json
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from collections import Counter

# TODO: -simplify a process of getting from atomfinder output to graph
#       where nodes and edges correspond to atoms and bonds
#       (it works fine now, but is a bit overcomplicated)
#       -save library data in USID format

def to_dataframe(coordinates, atoms):
    """
    Transforms output of atomfinder to panda dataframe
    
    Parameters:
    ----------
    coordinates: numpy array (dtype=np.float)
        array with shape of nrows*3
        first two columns are xy coordinates
        third column is atomic classes
    atoms: dict
        dictionary defining lattice and dopant atom types

    Returns
    -------
    df: pandas dataframe
        dataframe with atom type and coordinates
    """
    x, y, atomlist = coordinates.T
    atomlist = np.array(atomlist, dtype=np.str)
    atomlist[atomlist==str(0.)] = atoms['lattice_atom']
    atomlist[atomlist==str(1.)] = atoms['dopant']
    columns = ['atom', 'x', 'y']
    df = pd.DataFrame({'atom':atomlist, 'x':x, 'y':y}, columns = columns)
    return df

def make_graph_nodes(dataframe, verbose=True):
    """
    Creates graph nodes from a pandas dataframe with atomic coordinates
    
    Parameters
    ----------
    dataframe: pandas dataframe
        dataframe with atom types and positions
    verbose: boolean

    Returns
    -------
    U: networkx graph object (nodes only)
    u_atoms: unique atomic species in graph
    """
    u_atoms, indices = np.unique(dataframe.values[:,0], return_index=True)
    if verbose:
        print('Found the following atomic species:', ', '.join(u_atoms.tolist()))
    U=nx.Graph()
    for j in range(len(indices)):
        if j + 1 != len(indices):
            for i, (idx, x, y) in enumerate(dataframe.values[indices[j] : indices[j+1]]):
                U.add_node(idx+" {}".format(i+1), pos=(y, x))
        else:
            for i, (idx, x, y) in enumerate(dataframe.values[indices[j] : ]):
                U.add_node(idx+" {}".format(i+1), pos=(y, x))
    n_nodes = len(U.nodes())
    if verbose:
        print('Created', str(n_nodes), 'graph nodes corresponding to atomic species') 
    return U, u_atoms

def dist(U1, U2, p1, p2):
    """
    Calculates distances between nodes of a given graph(s)
    """
    return np.sqrt((U1.node[p1]['pos'][1]-U2.node[p2]['pos'][1])**2 + 
           (U1.node[p1]['pos'][0]-U2.node[p2]['pos'][0])**2)

def atomic_pairs_data(atomic_species, target_size, *args, **kwargs):
    """
    Creates dictionary of atomic pairs with
    a maximum allowed bond length for each pair
    """
    try:
        image_size, = kwargs.values()
    except ValueError:
        image_size = float(input("Enter the size of the image '" + str(filename_c) + "' in picometers:"))
    atomic_pairs =  list(set(tuple(sorted(a)) for a in itertools.product(atomic_species, repeat = 2)))
    atomic_pairs_dict = {}
    for pair in atomic_pairs:
        dictionary = {}
        if args:
            for pair_i in list(itertools.permutations(pair)):
                if pair_i in args[0].keys():
                    l = args[0][pair_i]
        else:
            print('Enter the maximum allowed bond length (in picometers) for a pair', str(pair) + ':')
            l = float(input())
        l_px = l*(target_size[0]/image_size)
        dictionary['atomic_pair_bond_length'] = l_px
        atomic_pairs_dict[pair] = dictionary
    return atomic_pairs_dict, image_size

def create_graph_edges(U, atomic_pairs_d):
    """
    Add edges to a graph ("chemical bonds")

    Parameters
    ----------
    U: networkx graph
        graph with nodes already defined
    atomic_pairs_d: dict
        dictionary defining maximum bond length allowed for each pair of atoms
    
    Returns
    -------
    U: networkx graph
        graph with nodes and edges
    """
    for k in atomic_pairs_d.keys():
        for p1 in U.nodes():
            for p2 in U.nodes():
                if all([(p1.split()[0] == k[0]),
                        (p2.split()[0] == k[1])]):
                    distance = dist(U, U, p1, p2)
                    if 0 < distance < atomic_pairs_d[k]['atomic_pair_bond_length']:
                        U.add_edge(p1, p2)
    return U

def refine_structure(U, *args, verbose=True, **kwargs):
    """
    Removes graph nodes (and corresponding edges)
    if they are not directly connected to a dopant
    
    Parameters
    ----------
    U: networkx graph
        graph with nodes and edges defined
    *args: dict
        dictionary defining lattice and dopant atom types
    **kwargs: int
        maximum coordination number for dopant
    
    Returns:
    -------
    U: networkx graph
        refined graph (dopant and its first coordination sphere)
    """
    try:
        lattice_atom, dopant = args[0].values()
    except IndexError:
        lattice_atom = str(input('\nEnter the name of lattice atoms (e.g. C, Ga, Si, etc):\n'))
        dopant = str(input('\nEnter the name of dopant (e.g. N, P, Si, etc):\n'))
    remove_l_edges = [edge for edge in U.edges() if dopant not in str(edge)]
    U.remove_edges_from(remove_l_edges)
    remove_lone_atoms = [node for node, degree in U.degree() if degree == 0 and node.split()[0] == lattice_atom]
    U.remove_nodes_from(remove_lone_atoms)
    if verbose:
        print('All lattice atoms not directly connected to a dopant have been removed')
    try:
        max_coord, = kwargs.values()
    except ValueError:
        max_coord = int(input('Enter the expected maximum coordination number for dopant atom:'))
    dopant_extra = [node for node, degree in U.degree() if degree > max_coord and node.split()[0] == dopant]
    for atom in dopant_extra:
        remove_dopant_edges = []
        while len(list(U.neighbors(atom))) > max_coord:
            neighbors = list(U.neighbors(atom))
            neighbors = [n for n in neighbors if n.split()[0] != dopant]
            Distance = np.array([])
            for nbs in neighbors:
                Dist = dist(U, U, nbs, atom)
                Distance = np.append(Distance, Dist)
            dist_max = np.unravel_index(np.argmax(Distance),Distance.shape)
            remove_dopant_edges.append(tuple((atom, neighbors[dist_max[0]])))
            U.remove_edges_from(remove_dopant_edges)
    remove_lone_atoms = [node for node, degree in U.degree() if degree == 0 and node.split()[0] == lattice_atom]
    U.remove_nodes_from(remove_lone_atoms)
    if verbose:
        print('Refinement procedure based on the maximum coordination number has been completed')

def plot_graph(Graph, atomic_species, exp_img, img_size,
               atomic_labels=True, overlay=False, **kwargs):
    """
    Plots experimental image and lattice graph
    
    Parameters:
    ----------
    Graph: networkx graph
        graph with defined nodes and edges
    atomic_species: list of strings
        list of unique atomic species
    exp_img: 2D or 4D numpy array
        experimental image data of shape width*height or
        nbatches*width*height*nchannels
    img_size: float
        size of image in picometers
    atomic_labels: boolean
        Show labels corresponding to atomic spcecies for each node
    overlay: boolean
        Plot graph on top of experimental image
    **node_size: float
        size of graph nodes
    **font_size: float
        size of node labels

    Returns:
    -------
    Graph plot
    """
    c_nodes = kwargs.get('node_size') if kwargs.get('node_size') is not None else 200
    c_fonts = kwargs.get('font_size') if kwargs.get('font_size') is not None else c_nodes/15
    if np.ndim(exp_img) == 4:
        exp_img = exp_img[0, :, :, 0]
    color_map = []
    colors = ['orange', 'red', 'blue', 'black', 'magenta', 'gray', 'green']
    for node in Graph.nodes():
        for i in range(len(atomic_species)):
            if node.split()[0] == atomic_species[i]:
                    color_map.append(colors[i])
    labels_atoms = {}
    for i in range (len(atomic_species)):
        for node in Graph.nodes():
            if node.split()[0] == atomic_species[i]:
                labels_atoms[node] = atomic_species[i]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[1].set_xlim(0, exp_img.shape[0])
    ax[1].set_ylim(exp_img.shape[1], 0)
    ax[0].axis('off')
    ax[0].set_aspect('equal')
    ax[0].set_title('Experimental image', fontsize=8)
    ax[0].imshow(exp_img, cmap='gray')
    alpha_ = 1
    edge_color_ = 'gray'
    if overlay == True:
        ax[1].imshow(exp_img, cmap='gray')
        alpha_ = 0.4
        edge_color_ = 'white'
    ax[1].set_title('Decoded', fontsize=8)
    ax[1].set_xlim(0, exp_img.shape[0])
    ax[1].set_ylim(exp_img.shape[1], 0)
    ax[1].axis('off')
    ax[1].set_aspect('equal')
    pos = nx.get_node_attributes(Graph, 'pos')
    nx.draw_networkx_nodes(Graph, pos=pos, nodelist=Graph.nodes(), node_color=color_map,
                           node_size=c_nodes, alpha=alpha_)
    nx.draw_networkx_edges(Graph, pos, width=1.5, edge_color=edge_color_, alpha=alpha_)
    if atomic_labels == True:
        nx.draw_networkx_labels(Graph, pos,labels=labels_atoms,font_size=c_fonts)
    elif atomic_labels == False:
        nx.draw_networkx_labels(Graph, pos,font_size=c_fonts/2)
    plt.show(block=False)

def get_subgraphs(Graph, verbose=True):
    """
    Finds individual defects after graph refinement procedure

    Parameters:
    ----------
    Graph: networkx graph
    verbose: boolean

    Returns:
    -------
    sub_graphs: generator
        generator of graphs (one for each connected component)
    """
    sub_graphs = list(nx.connected_component_subgraphs(Graph))
    if verbose:
        print('\nIdentified', len(sub_graphs), 'defect structures')
    return sub_graphs

def get_defect_coord(sg):
    """
    Returns coordinates of the center of the mass of a defect
    and all the coordinates and type of all the atoms in the defect

    Parameters
    ----------
    sg: networkx graph
        graph correponding to a isolated atomic defect structure

    Returns
    -------
    defect_com: list of floats
        center of the mass coordinates
    defect_atom_coord: numpy array
        array with nrows*3 shape; first two columns are xy coordinates,
        third column is atomic classes
    """
    defect_coord_x = np.array([])
    defect_coord_y = np.array([])
    defect_atom = []
    for d in sg.node:
        defect_atom.append(d.split(' ')[0])
        defect_coord_x = np.append(defect_coord_x, sg.node[d]['pos'][0])
        defect_coord_y = np.append(defect_coord_y, sg.node[d]['pos'][1])
    defect_atom = np.array(defect_atom)
    defect_atom_coord = np.concatenate((defect_coord_x[:, None],
                                        defect_coord_y[:, None],
                                        defect_atom[:, None]), axis=1)
    mean_x = np.around(np.mean(defect_coord_x), decimals=2)
    mean_y = np.around(np.mean(defect_coord_y), decimals=2)
    defect_com = [mean_x, mean_y]
    return defect_com, defect_atom_coord

def get_angles(sg, dopant):
    """
    Calculates angles around dopant(s) for a subgraph
    
    Parameters
    ----------
    sg: networkx graph
        graph corresponding to isolated atomic structure
    dopant: str
        name of dopant

    Returns
    -------
    angles: numpy array
        1D array of angles formed between dopant and host lattice atoms
    """
    angles = np.array([])
    for p1 in [node for node in sg.nodes() if node.split()[0] == dopant]:
        for atuple in list(set(tuple(sorted(a)) for a in itertools.product(sg.neighbors(p1), repeat = 2))):
            if atuple[0] != atuple[1]:
                points = [atuple[0], p1, atuple[1]]
                u = np.array(sg.node[points[1]]['pos']) - np.array(sg.node[points[0]]['pos'])
                v = np.array(sg.node[points[1]]['pos']) - np.array(sg.node[points[2]]['pos'])
                a = np.dot(u, v)
                b = np.linalg.norm(u) * np.linalg.norm(v)
                angles = np.append(angles, np.arccos(a/b) * 180 / np.pi)
    return angles

def get_bond_lengths(sg, dopant, img_size, exp_img):
    """
    Calculates bond length between dopant(s) and
    surrounding host lattice atoms for a subgraph
    
    Parameters
    ----------
    sg: networkx graph
        graph corresponding to isolated atomic structure
    dopant: str
        name of dopant
    img_size: float
        size of image in picometers
    exp_img: 2D or 4D numpy array
        experimental image with equal height and width
    
    Returns
    -------
    bond_length: 1D numpy array
        array of bond lengths formed between dopant and lattice atoms
    """
    bond_length = np.array([])
    for p1 in [node for node in sg.nodes() if node.split()[0] == dopant]:
        for p2 in sg.neighbors(p1):
            sc = img_size/exp_img.shape[1]
            bond_length = np.append(bond_length, dist(sg, sg, p1, p2)*sc)
    return np.unique(bond_length)

def construct_graphs(img, img_size, coord, atoms, approx_max_bonds, *args,
                     raw_data=True, save_all=False, plot_result=True, verbose=True):
    """
    Constructs graphs, plots them and saves defect coordinates with the image

    Parameters
    ----------
    img: 2D or 4D numpy array
        experimental image
    img_size: float
        experimental image size in picometers
    coord: numpy array
        atomic coordinates in a form of nrows*3 array; first two columns are
        xy coordinates, third column is atomic classes
    atoms: dict
        dictionary defining lattice and dopant atom types
    approx_max_bonds: dict
        dictionary defining maximum allowable bond lengths
        for each pair of atomic species
    *args: str
        filename (used when result is saved)
    raw_data: boolean
        Indicates whether the data is raw/new or was already processed
    save_all: boolean
        saves all processed data without asking
    plot_results: boolean
        plots constructed graphs
    verbose: boolean

    Returns
    -------
    Plots and saves image data with the coordinates of extracted atomic
    structures/defects
    """
    # Parsing *args list
    arg1 = [a for a in args if isinstance(a, str)]
    imgfile = arg1[0] if len(arg1) > 0 else None
    arg2 = [a for a in args if isinstance(a, dict)]
    metadata = arg2[0] if len(arg2) > 0 else None
    target_size = img.shape[1:3] if np.ndim(img) == 4 else img.shape
    df = to_dataframe(coord, atoms)
    U, atomic_species = make_graph_nodes(df, verbose)
    atomic_pairs_d, img_size = atomic_pairs_data(
        atomic_species, target_size, approx_max_bonds, image_size=img_size)
    create_graph_edges(U, atomic_pairs_d)
    refine_structure(U, atoms, verbose=verbose, max_coord=4)
    if plot_result:
        plot_graph(
            U, atomic_species, img,
            img_size, atomic_labels=True, overlay=True
        )
    sub_graphs = get_subgraphs(U, verbose)
    # Analyze each defect in the image
    for i, sg in enumerate(sub_graphs):
        atom_list = []
        for n in sg.nodes():
            atom_list.append(n.split(' ')[0])
        n_imp = Counter(atom_list)[atoms['dopant']]
        n_host = Counter(atom_list)[atoms['lattice_atom']]
        defect_formula = atoms['dopant'] + str(n_imp) + atoms['lattice_atom'] + str(n_host)
        defect_position, defect_coord = get_defect_coord(sg)
        if verbose:
            print('Defect {}:\n'.format(i+1),
                  'Defect formula:', defect_formula,
                  'Defect position:', defect_position)
        if not raw_data and i+1 < len(list(sub_graphs)):
            continue
        if not raw_data:
            return sub_graphs
        if save_all:
            save_option = 'Y'
        else:
            save_option = input(('Save data: [Y]es or [N]o\n'))
        if save_option == 'Y' or save_option == 'y':
            _filepath = 'library_test/' + defect_formula
            # Create a directory for each defect formula
            if not os.path.exists(_filepath):
                    os.makedirs(_filepath)
            # save hdf5 file with original and decoded data
            if imgfile is None:
                imgfile = input(('Please enter a name for hdf5 file to be stored on a disk'))
            _filename = os.path.splitext(imgfile)[0].split('/')[-1]+'.hdf5'
            with h5py.File(os.path.join(_filepath, _filename), 'a') as f:
                if 'nn_input' not in f.keys():
                    nn_input = f.create_dataset('nn_input', data=img)
                    nn_input.attrs['scan size'] = img_size
                    if metadata is not None:
                        nn_input.attrs['metadata'] = json.dumps(metadata)
                if 'defect_coord_{}'.format(i) not in f.keys():
                    coord_d = f.create_dataset('defect_coord_{}'.format(i),
                                             data=np.string_(defect_coord, encoding="utf-8"))
                    coord_d.attrs['defect type'] = defect_formula
                    if verbose:
                        print('Saved file with defect coordinates to disk\n')
