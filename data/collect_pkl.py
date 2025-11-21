import torch
import tree
from data.pdb_dataloader import PdbDataset
from omegaconf import OmegaConf
import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from openfold.utils.loss import between_residue_bond_loss,between_residue_clash_loss
violation_tolerance_factor=12
from data import utils as du
def save_pdb(xyz, pdb_out="out.pdb"):
    pdb_out=pdb_out
    ATOMS = ["N","CA","C","O"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k+1,ATOMS[k%4],"GLY","A",a+1,x,y,z,1,0)
        )
        k += 1
        if k % 4 == 0: a += 1
    out.close()
def analysis():
    # Load the dataset
    data = pd.read_csv('//media/junyu/DATA/scope_preprocessed/metadata.csv')

    # Check the first few entries to understand the data structure
    data_head = data.head()

    # Display basic statistics of the 'seq_len' column
    seq_len_stats = data['seq_len'].describe()

    # Plot a histogram of the 'seq_len' column
    plt.figure(figsize=(10, 6))
    plt.hist(data['seq_len'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of seq_len')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    histogram_path = 'seq_len_histogram.png'
    plt.savefig(histogram_path)
    plt.close()

    # Plot a density plot for the 'seq_len' column
    plt.figure(figsize=(10, 6))
    data['seq_len'].plot(kind='density', color='green')
    plt.title('Density Plot of seq_len')
    plt.xlabel('Sequence Length')
    plt.grid(True)
    density_plot_path = 'seq_len_density.png'
    plt.savefig(density_plot_path)
    plt.close()
def colletdata():
    config = OmegaConf.load("../configs/binder_design.yaml")

    dataset_cfg=config.data.dataset

    pdbs=PdbDataset(dataset_cfg=dataset_cfg,is_training=False)

    # 创建一个空列表来收集数据
    collected_data = []

    chinasmax=[]
    resmax = []
    # 遍历数据集
    for i in tqdm.tqdm(pdbs):
        if i is None:
            continue
        else:
            collected_data=collected_data+i

        # atom4_atom_exists=i['res_mask'].unsqueeze(-1).repeat(1,1,4)
        # Compute between residue backbone violations of bonds and angles.
        # connection_violations = between_residue_bond_loss(
        #     pred_atom_positions=i['bbatoms'],
        #     pred_atom_mask=atom4_atom_exists,
        #     residue_index=i["res_idx"],
        #     aatype=i["aatype"],
        #     tolerance_factor_soft=violation_tolerance_factor,
        #     tolerance_factor_hard=violation_tolerance_factor,
        # )
        # save_pdb(i[0]['bbatoms'].reshape(-1, 3), 'demo1.pdb')

        # for pdb in i:
        #     chinasmax.append(torch.max(pdb['chain_idx']))
        #     resmax.append(torch.max(pdb['res_idx']))



        # collected_data=collected_data+i
        # chinasmax.append(torch.max(i['chain_idx']))  # 将每个项的数据添加到列表中

    # print(max(chinasmax))
    # print(max(resmax))


    #将收集到的数据保存为 .pkl 文件
    name=   pdbs.pklname
    if len(collected_data)>0:
        print('has data ',len(collected_data))
        with open(name+".pkl", "wb") as file:
            pickle.dump(collected_data, file)
    else:
        print('no data')
    return name+"_collected.pkl"


def split_list(input_list, L):
    return [input_list[i:i+L] for i in range(0, len(input_list), L)]

def crop(minidx, maxidx, length=256):
    # Calculate the difference
    diff = maxidx - minidx

    # If difference is less than or equal to length, return minidx and maxidx
    if diff <= length - 1:
        return minidx, maxidx
    else:
        # Choose a random starting point between minidx and (maxidx - length + 1) inclusive
        start = np.random.randint(minidx, maxidx - length + 2)
        end = start + length - 1
        return start, end
def loadpkl(processed_file_path):
    from data import utils as du
    from openfold.data import data_transforms
    from data.interpolant import save_pdb

    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # Only take modeled residues.
    modeled_idx = processed_feats['modeled_idx']

    # del processed_feats['modeled_idx']
    # processed_feats = tree.map_structure(
    #     lambda x: x[min_idx:(max_idx+1)], processed_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask'])
    }
    # chain_feats = data_transforms.atom37_to_frames(chain_feats)
    # rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    # rotmats_1 = rigids_1.get_rots().get_rot_mats()
    # trans_1 = rigids_1.get_trans()
    res_idx = processed_feats['residue_index']
    chain_idx = processed_feats['chain_index']
    # chain_idx=chain_idx- np.min(chain_idx) + 1
    #
    NCAC = torch.tensor(processed_feats['atom_positions'])[..., :3, :]
    O = torch.tensor(processed_feats['atom_positions'])[..., 4, :].unsqueeze(-2)
    bbatoms = torch.cat((NCAC, O), dim=-2).float()

    feats = []
    if len(bbatoms) > 512:
        sub_modeled_idxs = split_list(modeled_idx, L=384)
        # to tensor
        res_mask = torch.tensor(processed_feats['bb_mask']).int()
        chain_idx = torch.tensor(chain_idx).int()
        res_idx = torch.tensor(res_idx).int()


        for sub_modeled_idx in sub_modeled_idxs:
            if len(sub_modeled_idx)>=60:
                min_idx = np.min(sub_modeled_idx)
                max_idx = np.max(sub_modeled_idx)

                sub_res_idx = res_idx[min_idx:(max_idx + 1)]
                sub_chain_idx = chain_idx[min_idx:(max_idx + 1)]

                subf = {
                    'aatype': chain_feats['aatype'][min_idx:(max_idx + 1)],
                    'res_idx': sub_res_idx - torch.min(sub_res_idx) + 1,
                    'bbatoms': bbatoms[min_idx:(max_idx + 1)],
                    'res_mask': res_mask[min_idx:(max_idx + 1)],
                    'chain_idx': sub_chain_idx - torch.min(sub_chain_idx) + 1,
                }
                feats.append(subf)
    save_pdb(feats[0]['bbatoms'].reshape(-1, 3), feats[0]['chain_idx'], 'demo1.pdb')

def checkloss():
    config = OmegaConf.load("../configs/base.yaml")

    dataset_cfg = config.data.dataset

    pdbs = PdbDataset(dataset_cfg=dataset_cfg, is_training=True)

    # 创建一个空列表来收集数据
    collected_data = []

    chinasmax = []
    # 遍历数据集

if __name__ == '__main__':
    #processed_feats1 = du.read_pkl('/media/junyu/DATA/mmcif/compedidx/hw/6hwn.pkl')
    # processed_feats2 = du.read_pkl('/media/junyu/DATA/rcsb_2.6_muilti_heteromer_2048_40_True.pkl')
    # processed_feats3 = du.read_pkl('/media/junyu/DATA/rcsb_2.6_muilti_homomer_2048_40_True.pkl')
    # collected_data=processed_feats1+processed_feats2+processed_feats3
    # name='homo_heto'
    # with open("//media/junyu/DATA/rcsb_"+name+".pkl", "wb") as file:
    #     pickle.dump(collected_data, file)

    # loadpkl('/media/junyu/DATA/rcsb_cluster/rcsb_homo_heto.pkl')
    colletdata()