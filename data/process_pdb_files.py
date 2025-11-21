"""Script for preprocessing PDB files."""

import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
from Bio import PDB
import numpy as np
import mdtraj as md


from data import utils as du
from data import parsers
from data import errors


# Define the parser
parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--pdb_dir',
    help='Path to directory with PDB files.',
    default='/home/junyu/project/Proflow/examples/sym_motif/nikel_4chains/',
    type=str)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=10)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default='/preprocessed/')
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')
parser.add_argument(
    '--output_ss',default=True,
    help='Whether to output SS sequence to ref_ss.txt.',
    action='store_true')


def process_file(file_path: str, write_dir: str, output_ss: bool = False):
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.
        output_ss: Whether to output SS sequence to ref_ss.txt.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path).replace('.pdb', '')
    metadata['pdb_name'] = pdb_name

    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)
    metadata['raw_path'] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # 获取头部信息，这里的头部信息包含了PDB文件中定义的各种元数据，包括COMPND
    header = parser.get_header()
    compnd_info = header.get("compound", {})



    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    # 遍历compnd_info，为每个复合物分配com_idx
    for com_idx, compnd in compnd_info.items():
        chains_in_compnd = compnd['chain'].upper().replace(" ", "").split(",")

        print(com_idx,'is',compnd)

        for chain_id in chains_in_compnd:
            chain = struct_chains[chain_id]  # 假设是第一个模型
            # Convert chain id into int
            chain_id = du.chain_str_to_int(chain_id)
            chain_prot = parsers.process_chain(chain, chain_id)
            chain_dict = dataclasses.asdict(chain_prot)

            # 为当前chain增加comp_idx信息
            comidx_null=np.ones(chain_dict['chain_index'].shape)
            chain_dict['com_idx'] = int(com_idx)*comidx_null

            # do not center at begin
            # chain_dict = du.parse_chain_feats(chain_dict)
            all_seqs.add(tuple(chain_dict['aatype']))
            struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx


############################chain mode##################################################

    # for chain_id, chain in struct_chains.items():
    #     # Convert chain id into int
    #     chain_id = du.chain_str_to_int(chain_id)
    #     chain_prot = parsers.process_chain(chain, chain_id)
    #     chain_dict = dataclasses.asdict(chain_prot)
    #     # do not center at begin
    #     # chain_dict = du.parse_chain_feats(chain_dict)
    #     all_seqs.add(tuple(chain_dict['aatype']))
    #     struct_feats.append(chain_dict)
    # if len(all_seqs) == 1:
    #     metadata['quaternary_category'] = 'homomer'
    # else:
    #     metadata['quaternary_category'] = 'heteromer'
    # complex_feats = du.concat_np_features(struct_feats, False)
    #
    # # Process geometry features
    # complex_aatype = complex_feats['aatype']
    # metadata['seq_len'] = len(complex_aatype)
    # modeled_idx = np.where(complex_aatype != 20)[0]
    # if np.sum(complex_aatype != 20) == 0:
    #     raise errors.LengthError('No modeled residues')
    # min_modeled_idx = np.min(modeled_idx)
    # max_modeled_idx = np.max(modeled_idx)
    # metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    # complex_feats['modeled_idx'] = modeled_idx
    ############################chain mode##################################################
    try:
        # MDtraj
        traj = md.load(file_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        # os.remove(file_path)
    except Exception as e:
        # os.remove(file_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]
    complex_feats['ss'] = pdb_ss[0]
    # Write features to pickles.
    du.write_pkl(processed_path, complex_feats)

    # Output SS sequence to txt file if requested
    if output_ss:
        from data.residue_constants import restypes
        ss_txt_path = os.path.join(write_dir, f'{pdb_name}_ref_ss.txt')

        # Convert aatype to sequence
        complex_aatype = complex_feats['aatype']
        modeled_idx = complex_feats['modeled_idx']

        # Get modeled residues only
        modeled_aatype = complex_aatype[modeled_idx]
        modeled_ss = pdb_ss[0][modeled_idx]

        # Convert aatype numbers to letters
        sequence = ''.join([restypes[aa] if aa < 20 else 'X' for aa in modeled_aatype])
        ss_sequence = ''.join(modeled_ss)

        # Write to file in FASTA-like format
        with open(ss_txt_path, 'w') as f:
            f.write(f'>{pdb_name}_sequence\n')
            f.write(f'{sequence}\n')
            f.write(f'>{pdb_name}_secondary_structure\n')
            f.write(f'{ss_sequence}\n')

    # Return metadata
    return metadata


def process_genpdb(file_path: str, write_dir: str):
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path).replace('.pdb', '')
    metadata['pdb_name'] = pdb_name

    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)
    metadata['raw_path'] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)


    # del struct_chains['C']
    # del struct_chains['D']
    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx

    try:
        # MDtraj
        traj = md.load(file_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        # os.remove(file_path)
    except Exception as e:
        # os.remove(file_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0][:metadata['modeled_seq_len']]
    metadata['coil_percent'] = np.sum(chain_dict['ss'] == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(chain_dict['ss'] == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(chain_dict['ss'] == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]



    # Return metadata
    return metadata,complex_feats

def process_serially(all_paths, write_dir, output_ss=False):
    all_metadata = []
    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            metadata = process_file(
                file_path,
                write_dir,
                output_ss=output_ss)
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')
    return all_metadata


def process_fn(
        file_path,
        verbose=None,
        write_dir=None,
        output_ss=False):
    try:
        start_time = time.time()
        metadata = process_file(
            file_path,
            write_dir,
            output_ss=output_ss)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


def main(args):
    pdb_dir = args.pdb_dir
    all_file_paths = [
        os.path.join(pdb_dir, x)
        for x in os.listdir(args.pdb_dir) if '.pdb' in x]
    total_num_paths = len(all_file_paths)
    write_dir = pdb_dir+args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths,
            write_dir,
            output_ss=args.output_ss)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir,
            output_ss=args.output_ss)
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, all_file_paths)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)
    '''
    this is only suitable for native because generated pdn has not compund info
    '''
