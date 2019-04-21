"""
Utility functions for constructing lattice graphs

Author: Maxim Ziatdinov
"""
import os
import h5py
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from collections import Counter


def to_dataframe(coordinates, atoms):
    """
    Transforms output of atomfinder to a panda dataframe
    """
    x, y, atomlist = coordinates.T
    atomlist = np.array(atomlist, dtype=np.str)
    atomlist[atomlist==str(0.)] = atoms['lattice_atom']
    atomlist[atomlist==str(1.)] = atoms['dopant']
    columns = ['atom', 'x', 'y']
    df = pd.DataFrame({'atom':atomlist, 'x':x, 'y':y}, columns = columns)
    return df
   
def make_graph_nodes(dataframe):
    """
    Creates graph nodes from a panda dataframe with atomic coordinates
    More details TBA
    """
    u_atoms, indices = np.unique(dataframe.values[:,0], return_index=True)
    print('Found the following atomic species:', ', '.join(u_atoms.tolist()))
    U=nx.Graph()
    for j in range(len(indices)):
        if j + 1 != len(indices):
            for i, (idx, x, y) in enumerate(dataframe.values[indices[j] : indices[j+1]]):
                U.add_node(idx+" {}".format(i+1), pos=(y, x))
        else:
            for i, (idx, x, y) in enumerate(dataframe.values[indices[j] : ]):
                U.add_node(idx+" {}".format(i+1), pos=(y, x))
    pos = nx.get_node_attributes(U,'pos')
    n_nodes = len(U.nodes())
    print('Created', str(n_nodes), 'graph nodes corresponding to atomic species') 
    return U, n_nodes, pos, u_atoms

def dist(U1, U2, p1, p2):
    """
    Calculates distances between nodes of a given graph(s)
    """
    return np.sqrt((U1.node[p1]['pos'][1]-U2.node[p2]['pos'][1])**2 + 
           (U1.node[p1]['pos'][0]-U2.node[p2]['pos'][0])**2)

def atomic_pairs_data(atomic_species, target_size, *args, **kwargs):
    '''Creates dictionary of atomic pairs with
       a maximum allowed bond length for each pair'''
    try:
        image_size, = kwargs.values()
    except:
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

def refine_structure(U, *args, **kwargs):
    """
    Removes graph nodes (and corresponding edges) if they are not
    directly connected to the dopant
    More details TBA
    """
    try:
        lattice_atom, dopant = args[0].values()
    except:
        lattice_atom = str(input('\nEnter the name of lattice atoms (e.g. C, Ga, Si, etc):\n'))
        dopant = str(input('\nEnter the name of dopant (e.g. N, P, Si, etc):\n'))
    remove_l_edges = [edge for edge in U.edges() if dopant not in str(edge)]
    U.remove_edges_from(remove_l_edges)
    remove_lone_atoms = [node for node, degree in U.degree() if degree == 0 and node.split()[0] == lattice_atom]
    U.remove_nodes_from(remove_lone_atoms)
    print('All lattice atoms not directly connected to a dopant have been removed')
    try:
        max_coord, = kwargs.values()
    except:
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
    print('Refinement procedure based on the maximum coordination number has been completed')

def plot_graph(Graph, atomic_species, pos, exp_img, img_size,
               atomic_labels=True, overlay=False, **kwargs):
    """
    Plots experimental image and lattice graph
    More details TBA
    """
    c_nodes = kwargs.get('node_size') if kwargs.get('node_size') is not None else 200
    c_fonts = kwargs.get('font_size') if kwargs.get('font_size') is not None else c_nodes/15
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
    nx.draw_networkx_nodes(Graph, pos=pos, nodelist=Graph.nodes(), node_color=color_map,
                           node_size=c_nodes, alpha=alpha_)
    nx.draw_networkx_edges(Graph, pos,width=1.5, edge_color=edge_color_, alpha=alpha_)
    if atomic_labels == True:
        nx.draw_networkx_labels(Graph, pos,labels=labels_atoms,font_size=c_fonts)
    elif atomic_labels == False:
        nx.draw_networkx_labels(Graph, pos,font_size=c_fonts/2)
    plt.show(block=False)
    
def get_subgraphs(U):
    """
    Finds individual defects after graph refinement procedure
    More details TBA
    """
    sub_graphs = list(nx.connected_component_subgraphs(U))
    print('\nIdentified', len(sub_graphs), 'defect structures')
    return sub_graphs

def get_defect_coord(sg):
    """
    Returns coordinates of the center of the mass of a defect
    and all the coordinates and type of all the atoms in the defect
    More details TBA
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
    bond_length = np.array([])
    for p1 in [node for node in sg.nodes() if node.split()[0] == dopant]:
        for p2 in sg.neighbors(p1):
            sc = img_size/exp_img.shape[1]
            bond_length = np.append(bond_length, dist(sg, sg, p1, p2)*sc)
    return bond_length

def construct_graphs(img, img_size, coord, atoms, approx_max_bonds,
                     *args, raw_data=True, save_all=False, plot_result=True):
    """
    Constructs graphs, plots them and saves defect coordinates with the image
    More details TBA
    """
    try:
        imgfile = args[0]
    except IndexError:
        imgfile = None
    target_size = img.shape[1:3]
    df = to_dataframe(coord, atoms)
    U, n_nodes, pos, atomic_species = make_graph_nodes(df)
    atomic_pairs_d, image_size=atomic_pairs_data(
        atomic_species, target_size, approx_max_bonds, image_size=img_size)
    create_graph_edges(U, atomic_pairs_d)
    refine_structure(U, atoms, max_coord=4)
    if plot_result:
        plot_graph(
            U, atomic_species, pos, img,
            img_size, atomic_labels=True, overlay=True
        )
    sub_graphs = get_subgraphs(U)
    # Analyze each defect in the image
    for i, sg in enumerate(sub_graphs):
        atom_list = []
        for n in sg.nodes():
            atom_list.append(n.split(' ')[0])
        n_imp = Counter(atom_list)[atoms['dopant']]
        n_host = Counter(atom_list)[atoms['lattice_atom']]
        defect_formula = atoms['dopant'] + str(n_imp) + atoms['lattice_atom'] + str(n_host)
        defect_position, defect_coord = get_defect_coord(sg)
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
                if 'defect_coord_{}'.format(i) not in f.keys():
                    f.create_dataset('defect_coord_{}'.format(i),
                                     data=np.string_(defect_coord, encoding="utf-8"))
            print('Saved file with defect coordinates to disk\n')
