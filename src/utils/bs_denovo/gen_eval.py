from . import bs_chem
import numpy as np

def standard_metrics(gen_txt_list, trn_set:set, subs_size, n_jobs=1):
    """
    Compute standard evaluation metrics for a list of generated SMILES strings.

    The metrics include:
      - **validity**: fraction of valid SMILES
      - **uniqueness**: fraction of unique SMILES among valid ones
      - **novelty**: fraction of generated compounds not present in training set
      - **internal diversity (intdiv)**: average pairwise dissimilarity (1 - Tanimoto)

    :param gen_txt_list: List of generated molecule SMILES strings.
    :param trn_set: Set of training SMILES strings (canonical).
    :param subs_size: Number of samples to use when computing internal diversity.
                      The first `subs_size` canonical SMILES are used.
    :param n_jobs: Number of parallel processes for fingerprint computation and similarity.
    :return: Dictionary containing metric names and their values.
             Keys: 'validity', 'uniqueness', 'novelty', 'intdiv'
    """
    std_mets = {}
    gsize = len(gen_txt_list)
    can_smis, invids = bs_chem.get_valid_canons(gen_txt_list, n_jobs)

    std_mets['validity'] = len(can_smis) / gsize
    uni_smis = list(set(can_smis))
    if len(uni_smis) <= 0:
        std_mets['uniqueness'] = -1
        std_mets['novelty'] = -1
        std_mets['intdiv'] = -1
    else:
        std_mets['uniqueness'] = len(uni_smis) / len(can_smis)
    
        gen_set = set(uni_smis)
        nov_set = gen_set.difference(trn_set)
        std_mets['novelty'] = len(nov_set) / len(uni_smis)

        subs = can_smis[:subs_size]
        sub_fps = bs_chem.get_mgfps_from_smilist(subs)  # use default options
        simmat = bs_chem.get_pw_simmat(sub_fps, sub_fps, sim_tup=bs_chem.tansim_tup, 
                                    n_jobs=n_jobs)  # using tanimoto similarity
        std_mets['intdiv'] = (1-simmat).mean()
    return std_mets
