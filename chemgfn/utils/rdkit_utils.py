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

try:
    # doesnt work for github CI/pypi install.
    # SA score is only in the contrib dir for RDKit conda install
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    sys.path.append(rdkit.__path__[0])
    import sascorer
except:
    # download SA score
    import requests

    files = [
        "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py",
        "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz",
    ]

    for file in files:
        r = requests.get(file)
        with open(rdkit.__path__[0] + "/" + file.split("/")[-1], "wb") as f:
            f.write(r.content)

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


FUNCTION_MAPPING = {
    "sa": sa_scorer,
    "logP": logP,
}
