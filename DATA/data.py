import itertools
import random
import sys
import time
from pathlib import Path
from typing import Optional

import os
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms

from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
from jarvis.db.figshare import data as jdata
from jarvis.core.specie import chem_data, get_node_attributes

# from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson

from pandarallel import pandarallel
import periodictable
import algorithm
import dgl
import json
import zipfile
from geometric_computing import xyz_to_dat

pandarallel.initialize(progress_bar=True)

tqdm.pandas()

torch.set_printoptions(precision=10)


def find_index_array(A, B):
    _, n = B.shape
    index_array = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        idx = torch.where((A == B[:, i].unsqueeze(1)).all(dim=0))[0]
        index_array[i] = idx

    return index_array

class StructureDataset(InMemoryDataset):

    def __init__(
        self,
        df,
        data_path,
        processdir,
        target,
        name,
        atom_features="atomic_number",
        id_tag="jid",
        root='./',
        transform=None,
        pre_transform=None,
        pre_filter=None,
        mean=None, std=None, normalize=False
    ):

        self.df = df
        self.data_path = data_path
        self.processdir = processdir
        self.target = target
        self.name = name
        self.atom_features = atom_features
        self.id_tag = id_tag
        self.ids = self.df[self.id_tag]
        self.labels = torch.tensor(self.df[self.target]).type(
            torch.get_default_dtype()
        )

        if mean is not None:
            self.mean = mean
        elif normalize:
            self.mean = torch.mean(self.labels)
        else:
            self.mean = 0.0

        if std is not None:
            self.std = std
        elif normalize:
            self.std = torch.std(self.labels)
        else:
            self.std = 1.0

        self.group_id = {
            "H": 0,
            "He": 1,
            "Li": 2,
            "Be": 3,
            "B": 4,
            "C": 0,
            "N": 0,
            "O": 0,
            "F": 5,
            "Ne": 1,
            "Na": 2,
            "Mg": 3,
            "Al": 6,
            "Si": 4,
            "P": 0,
            "S": 0,
            "Cl": 5,
            "Ar": 1,
            "K": 2,
            "Ca": 3,
            "Sc": 7,
            "Ti": 7,
            "V": 7,
            "Cr": 7,
            "Mn": 7,
            "Fe": 7,
            "Co": 7,
            "Ni": 7,
            "Cu": 7,
            "Zn": 7,
            "Ga": 6,
            "Ge": 4,
            "As": 4,
            "Se": 0,
            "Br": 5,
            "Kr": 1,
            "Rb": 2,
            "Sr": 3,
            "Y": 7,
            "Zr": 7,
            "Nb": 7,
            "Mo": 7,
            "Tc": 7,
            "Ru": 7,
            "Rh": 7,
            "Pd": 7,
            "Ag": 7,
            "Cd": 7,
            "In": 6,
            "Sn": 6,
            "Sb": 4,
            "Te": 4,
            "I": 5,
            "Xe": 1,
            "Cs": 2,
            "Ba": 3,
            "La": 8,
            "Ce": 8,
            "Pr": 8,
            "Nd": 8,
            "Pm": 8,
            "Sm": 8,
            "Eu": 8,
            "Gd": 8,
            "Tb": 8,
            "Dy": 8,
            "Ho": 8,
            "Er": 8,
            "Tm": 8,
            "Yb": 8,
            "Lu": 8,
            "Hf": 7,
            "Ta": 7,
            "W": 7,
            "Re": 7,
            "Os": 7,
            "Ir": 7,
            "Pt": 7,
            "Au": 7,
            "Hg": 7,
            "Tl": 6,
            "Pb": 6,
            "Bi": 6,
            "Po": 4,
            "At": 5,
            "Rn": 1,
            "Fr": 2,
            "Ra": 3,
            "Ac": 9,
            "Th": 9,
            "Pa": 9,
            "U": 9,
            "Np": 9,
            "Pu": 9,
            "Am": 9,
            "Cm": 9,
            "Bk": 9,
            "Cf": 9,
            "Es": 9,
            "Fm": 9,
            "Md": 9,
            "No": 9,
            "Lr": 9,
            "Rf": 7,
            "Db": 7,
            "Sg": 7,
            "Bh": 7,
            "Hs": 7
        }

        super(StructureDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.path.join(self.root, self.data_path)  # data_path是cache中的bin文件路径

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processdir)

    @property
    def processed_file_names(self):
        return self.name + '.pt'

    def process(self):
        mat_data = torch.load(self.raw_file_names)  # 加载cache中的缓存数据

        data_list = []
        features = self._get_attribute_lookup(self.atom_features)   #创建了一个特征查找表，可以根据元素的原子序数（Z）快速检索其对应的特征向量。

        for i in tqdm(range(len(mat_data))):
            node_features = mat_data[i].x
            mat_data[i].atom_numbers = node_features

            group_feats = []
            for atom in node_features:
                pe = periodictable.elements[int(atom)]
                e = pe.symbol
                group_feats.append(self.group_id[e])
            group_feats = torch.tensor(np.array(group_feats)).type(torch.LongTensor)
            identity_matrix = torch.eye(10) # 生成10阶单位矩阵
            g_feats = identity_matrix[group_feats]      # 包含原子的分组信息
            if len(list(g_feats.size())) == 1:
                g_feats = g_feats.unsqueeze(0)

            number_t = mat_data[i].atom_numbers.long()
            number_t = number_t.squeeze(1)
            f = features[number_t]
            f = torch.tensor(f).type(torch.FloatTensor)     # f 是根据原子序数在特征查找表中获得的特征
            if len(mat_data[i].atom_numbers) == 1:
                f = f.unsqueeze(0)

            mat_data[i].x = f
            mat_data[i].g_feats = g_feats

            y = (self.labels[i] - self.mean) / self.std     # 归一化目标值
            mat_data[i].y = y
            label_i = self.labels[i]        # 第i个数据对应的目标值
            mat_data[i].label = label_i

            data_list.append(mat_data[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        mat_data, slices = self.collate(data_list) # 将多个数据对象合并成一个批处理对象
        print("collate_data:\n", mat_data)
        print("co_slices:\n", slices)

        print('Saving...')
        # pp = self.processed_paths[0]
        torch.save((mat_data, slices), self.processed_paths[0])

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        max_z = max(v["Z"] for v in chem_data.values())

        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))
        # 初始化一个零矩阵，其维度为 (1 + max_z, len(template))。行数为 1 + max_z，确保每个可能的原子序数都有一行；列数为 template 的长度，对应特征向量的长度。

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features


def load_radius_graphs(
        df: pd.DataFrame,
        name: str = "dft_3d",
        target: str = "",
        radius: float = 4.0,
        max_neighbors: int = 16,
        cachedir: Optional[Path] = None,
):
    def atoms_to_graph(atoms):
        structure = Atoms.from_dict(atoms)
        sps_features = []
        for ii, s in enumerate(structure.elements):
            feat = list(get_node_attributes(s, atom_features="atomic_number"))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        edges = nearest_neighbor_edges(atoms=structure, cutoff=radius, max_neighbors=max_neighbors)
        u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)

        data = Data(x=node_features, edge_index=torch.stack([u, v]), edge_attr=r.norm(dim=-1))
        return data

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{target}-radius.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        pass
    else:
        graphs = df["atoms"].parallel_apply(atoms_to_graph).values
        torch.save(graphs, cachefile)


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))
    # print (r1,r1.shape)
    # print (r2,r2.shape)
    # print (bond_cosine,bond_cosine.shape)
    return {"h": bond_cosine}

def load_infinite_graphs(
        df: pd.DataFrame,
        name: str = "dft_3d",
        target: str = "",
        cachedir: Optional[Path] = Path('cache'),
        infinite_funcs=[],
        infinite_params=[],
        R=5,
):
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)      # 将字典数据转化为Atoms对象

        # build up atom attribute tensor
        sps_features = []
        for _, s in enumerate(structure.elements):
            fe = get_node_attributes(s, atom_features="atomic_number")
            feat = list(fe)
            sps_features.append(feat)

        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )

        inf_u = torch.arange(0, node_features.size(0), 1).unsqueeze(1).repeat((1, node_features.size(0))).flatten().long()
        inf_v = torch.arange(0, node_features.size(0), 1).unsqueeze(0).repeat((node_features.size(0), 1)).flatten().long()

        edge_index = torch.stack([inf_u, inf_v])

        lattice_mat = structure.lattice_mat.astype(dtype=np.double)

        u_t = inf_u.flatten().numpy().astype(np.int)
        v_t = inf_v.flatten().numpy().astype(np.int)
        vecs = structure.cart_coords[u_t] - structure.cart_coords[v_t]

        inf_edge_attr = torch.FloatTensor(
            np.stack(
                [
                    getattr(algorithm, func)(vecs, lattice_mat, param=param, R=R)
                    for func, param in zip(infinite_funcs, infinite_params)
                ],
            1
            )
        )

        edges = nearest_neighbor_edges(atoms=structure, cutoff=4, max_neighbors=16)
        u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)
        uv_edge = torch.stack([u, v])
        edge_attr = r.norm(dim=-1)

        edges_a = nearest_neighbor_edges(atoms=structure, cutoff=2, max_neighbors=8)
        u_a, v_a, r_a = build_undirected_edgedata(atoms=structure, edges=edges_a)
        edge_attr_a = r_a.norm(dim=-1)
        edge_index_a = torch.stack([u_a, v_a])
        pos = torch.tensor(structure.cart_coords)
        num_nodes = pos.size(0)
        dist, angle, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index_a, num_nodes, use_torsion=False)
        angle_index = torch.stack([idx_kj, idx_ji])

        data = Data(x=node_features, edge_attr=edge_attr, edge_index=uv_edge, inf_edge_index=edge_index, inf_edge_attr=inf_edge_attr, angle_attr=angle, angle_index=angle_index, edge_attr_a=edge_attr_a, edge_index_a=edge_index_a)

        return data

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{target}-infinite.bin"      # e.g. dft_3d_train-formation_energy_peratom-infinite.bin
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        # 如果已经有缓存数据文件，则不用重复构建
        pass 
    else:
        graphs = df["atoms"].parallel_apply(atoms_to_graph)     #使用 parallel_apply 方法并行地对 atoms 列中的每一个原子结构应用 atoms_to_graph 函数
        graphs = graphs.values
        torch.save(graphs, cachefile)


def get_id_train_val_test(
        total_size=1000,
        split_seed=123,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        n_train=None,
        n_test=None,
        n_val=None,
        keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
            train_ratio is None
            and val_ratio is not None
            and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1    # 确保不会超过一，否侧弹出异常

    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))   # 生成一个包含1~total_size-1的列表
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)             # 按照给定的随机数种子进行随机打乱ids

    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test): -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


# 输入数据列表
def get_torch_dataset(
        dataset=None,
        root="",
        cachedir="",
        processdir="",
        name="",
        id_tag="jid",
        target="",
        atom_features="",
        normalize=False,
        euclidean=False,
        cutoff=4.0,
        max_neighbors=16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
        mean=0.0,
        std=1.0,
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    vals = df[target].values
    print("data range", np.max(vals), np.min(vals))

    # 构建缓冲目录
    cache = os.path.join(root, cachedir)
    if not os.path.exists(cache):
        os.makedirs(cache)

    if euclidean:
        load_radius_graphs(
            df,
            radius=cutoff,
            max_neighbors=max_neighbors,
            name=name + "-" + str(cutoff),
            target=target,
            cachedir=Path(cache),
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{cutoff}-{target}-radius.bin"),
            processdir,
            target=target,
            name=f"{name}-{cutoff}-{target}-radius",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    else:   # 执行该支线
        load_infinite_graphs(
            df,
            name=name,
            target=target,
            cachedir=Path(cache),
            infinite_funcs=infinite_funcs,
            infinite_params=infinite_params,
            R=R,
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{target}-infinite.bin"),
            processdir,
            target=target,
            name=f"{name}-{target}-infinite",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    return data


def get_train_val_loaders(
        dataset: str = "dft_3d",
        root: str = "",
        cachedir: str = "",
        processdir: str = "",
        dataset_array=None,
        target: str = "formation_energy_peratom",
        atom_features: str = "cgcnn",
        n_train=None,
        n_val=None,
        n_test=None,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size: int = 64,
        split_seed: int = 123,
        keep_data_order=False,
        workers: int = 4,
        pin_memory: bool = True,
        id_tag: str = "jid",
        normalize=False,
        euclidean=False,
        cutoff: float = 4.0,
        max_neighbors: int = 16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
):
    if not dataset_array:
        print("###dataset:", dataset)
        d = jdata(dataset)
        # path = "/home/u2023170648/PotNet/jdft_test_big.json.zip"
        # js_tag = "jdft_test_big.json"
        # d = json.loads(zipfile.ZipFile(path).read(js_tag))
    else:
        d = dataset_array

    dat = []
    all_targets = []

    for i in d:
        # target是指预测的目标，比如formation_energy_peratom
        if isinstance(i[target], list):     # 判断i[target] 是一个列表（list）
            all_targets.append(torch.tensor(i[target]))
            dat.append(i)

        elif (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
        ):  # 输出查看得i[target]是一个数值，应该执行该分支
            dat.append(i)
            all_targets.append(i[target])

    # 根据比例获取训练，验证，测试集id
    id_train, id_val, id_test = get_id_train_val_test(
        total_size=len(dat),
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        keep_data_order=keep_data_order,
    )

    ids_train_val_test = {}  # 用来存储三个数据集对应的jid的值
    ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]  # dat是列表，i是id_train中的0~total_size分配到train中的序列，然后jid，就可以取到JVASP-***
    ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
    ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
    dumpjson(
        data=ids_train_val_test,
        filename=os.path.join(root, "ids_train_val_test.json"),
    ) # 对应输出目录中的ids_train_val_test.json文件

    dataset_train = [dat[x] for x in id_train]
    dataset_val = [dat[x] for x in id_val]
    dataset_test = [dat[x] for x in id_test]

    # print('using mp bulk dataset')
    # with open('/data/kruskallin/bulk_megnet_train.pkl', 'rb') as f:
    #     dataset_train = pk.load(f)
    # with open('/data/kruskallin/bulk_megnet_val.pkl', 'rb') as f:
    #     dataset_val = pk.load(f)
    # with open('/data/kruskallin/bulk_megnet_test.pkl', 'rb') as f:
    #     dataset_test = pk.load(f)
    #
    # target = 'bulk modulus'

    # print('using mp shear dataset')
    # with open('/data/kruskallin/shear_megnet_train.pkl', 'rb') as f:
    #     dataset_train = pk.load(f)
    # with open('/data/kruskallin/shear_megnet_val.pkl', 'rb') as f:
    #     dataset_val = pk.load(f)
    # with open('/data/kruskallin/shear_megnet_test.pkl', 'rb') as f:
    #     dataset_test = pk.load(f)
    # target = 'shear modulus'


    start = time.time()
    train_data = get_torch_dataset(
        dataset=dataset_train,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_train",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
    )

    mean = train_data.mean
    std = train_data.std

    val_data = get_torch_dataset(
        dataset=dataset_val,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_val",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
        mean=mean,
        std=std
    )

    test_data = get_torch_dataset(
        dataset=dataset_test,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_test",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
        mean=mean,
        std=std,
    )


    print("------processing time------: " + str(time.time() - start))   # 构件图结构所花的时间

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        mean,
        std,
    )
