import numpy as np

all_strings = np.load("/home/xw3763/project/data/smiles.npy")

from rdkit import Chem


def contains_cl_or_br(mol):
    """检查分子中是否含有氯或溴元素"""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in {"Cl", "Br"}:
            return True
    return False


# 或者使用更简洁的写法：
def contains_cl_or_br(mol):
    return any(atom.GetSymbol() in {"Br"} for atom in mol.GetAtoms())


from tqdm import tqdm

from chemgfn.utils.rdkit_utils import sa_scorer

all_frags = []
new_smiles = []
for i in tqdm(range(0, len(all_strings))):
    string = str(all_strings[i])
    mol = Chem.MolFromSmiles(string)
    try:
        sate = contains_cl_or_br(mol)
        new_smiles.append(string)
        if sate:
            if (sa_scorer(mol) > 3.0) and (len(string) <= 50):
                all_frags.append(string)
    except:
        continue

import torch

print(len(all_frags))
torch.save(all_frags, "/home/xw3763/project/data/smiles_3.0_len50.pt")
