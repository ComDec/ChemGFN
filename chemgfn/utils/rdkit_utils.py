import os
import shutil
import sys
from pathlib import Path

import partialsmiles as pa
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    QED,
    AllChem,
    Descriptors,
    Draw,
    PropertyMol,
    RDConfig,
    rdMMPA,
    rdMolDescriptors,
)
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

try:
    # doesnt work for github CI/pypi install.
    # SA score is only in the contrib dir for RDKit conda install
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    sys.path.append(rdkit.__path__[0])
    import sascorer
except:
    # download SA score with caching
    import requests

    # 使用缓存目录存储下载的文件
    cache_dir = Path.home() / ".cache" / "chemgfn" / "sa_score"
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py",
        "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz",
    ]

    for file_url in files:
        filename = file_url.split("/")[-1]
        cache_path = cache_dir / filename
        target_path = Path(rdkit.__path__[0]) / filename

        # 如果缓存文件存在，直接使用
        if cache_path.exists():
            if not target_path.exists() or target_path.stat().st_size != cache_path.stat().st_size:
                shutil.copy2(cache_path, target_path)
        else:
            # 下载文件并显示进度条
            try:
                response = requests.get(file_url, stream=True)
                response.raise_for_status()  # 检查HTTP错误
                total_size = int(response.headers.get("content-length", 0))

                with open(cache_path, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {filename}",
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                # 复制到目标目录
                shutil.copy2(cache_path, target_path)
            except Exception as e:
                # 如果下载失败，尝试删除不完整的缓存文件
                if cache_path.exists():
                    cache_path.unlink()
                raise RuntimeError(f"Failed to download {filename}: {e}") from e

    sys.path.append(rdkit.__path__[0])

    import sascorer


def verify_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                return True
            except:
                return False
        else:
            return False
    except:
        return False


def verify_smiles_pa(smiles: str):
    try:
        pa.ParseSmiles(smiles)
        return True
    except Exception:
        return False


def sa_scorer(mol):
    """
    Calculate the synthetic accessibility score of a molecule.
    """
    if mol is None:
        return 0
    try:
        return sascorer.calculateScore(mol)
    except:
        return 0


def logP(mol):
    """
    Calculate the logP of a molecule.
    """
    if mol is None:
        return -1
    try:
        return Descriptors.MolLogP(mol)
    except:
        return -1


def logP_drug(mol):
    """
    Calculate the logP of a drug molecule, optimal range: 0-3.
    """
    if mol is None:
        return -1
    try:
        return Descriptors.MolLogP(mol) if 0 <= Descriptors.MolLogP(mol) <= 3 else -1
    except:
        return -1


def logP_relative(mol):
    """
    Calculate the logP of a molecule.
    """
    if mol is None:
        return -1
    try:
        return Descriptors.MolLogP(mol) - 0.8561999999999999
    except:
        return -1


FUNCTION_MAPPING = {
    "sa": sa_scorer,
    "logP": logP,
    "logP_drug": logP_drug,
    "logP_relative": logP_relative,
}
