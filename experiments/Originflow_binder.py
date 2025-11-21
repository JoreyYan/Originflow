import os
import torch
import torch.nn as nn
import hydra
import tqdm
import re
from chroma.data.protein import Protein
from typing import Optional
from data.interpolant import Interpolant_10
from models.flow_model import FlowModel_binder, FlowModel_binder_sidechain,FlowModel_seqdesign
from data import utils as du
from data.pdb_dataloader import sPdbDataset
import numpy as np
import random
import datetime
from utils import set_global_seed,save_cfg
from data.utils import batch_align_structures,align_pred_to_true
from data.pdb_dataloader import calculate_interface_residues_vtarget



class Proflow(nn.Module):

    def __init__(
        self,
        cfg,
        weights_backbone: str = "named:public",
        weights_design: str = "named:public",
        device: Optional[str] = None,
        strict: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        import warnings

        warnings.filterwarnings("ignore")

        # If no device is explicity specified automatically set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self._exp_cfg=cfg.experiment
        self._inf_cfg=cfg.inference

        self._model = FlowModel_binder(cfg.model).to(self.device)




        self._load_from_state_dict()
        # self._load_from_state_dict_forsidechain()

        self._model.eval()
        self.Proflow = Interpolant_10(cfg.interpolant)
        self.sample_step=cfg.interpolant.sampling.num_timesteps

    def _load_from_state_dict(self, ) -> None:
        print('load from state dict')
        checkpoint = torch.load( self._inf_cfg.ckpt_path)
        original_state_dict = checkpoint['state_dict']
        # 加载权重之前，先调整键名
        adjusted_state_dict = {k[len("model."):]: v for k, v in original_state_dict.items() if k.startswith("model.")}

        # 创建更新后的模型实例
        # 修改原始状态字典以适应新模型（如果需要）
        # 例如，删除不再存在的层的权重或重命名某些键
        # updated_state_dict = {key: value for key, value in adjusted_state_dict.items() if
        #                       key in self._model.state_dict()}

        # 加载匹配的权重到新模型
        self._model.load_state_dict(adjusted_state_dict, strict=True)
        print('load from state dict finished')


    def _load_from_state_dict_forsidechain(self, ) -> None:
        print('load from state dict')
        checkpoint = torch.load( self._inf_cfg.sidechain_path)
        original_state_dict = checkpoint['state_dict']
        # 加载权重之前，先调整键名
        adjusted_state_dict = {k[len("model."):]: v for k, v in original_state_dict.items() if k.startswith("model.")}

        # 创建更新后的模型实例
        # 修改原始状态字典以适应新模型（如果需要）
        # 例如，删除不再存在的层的权重或重命名某些键
        # updated_state_dict = {key: value for key, value in adjusted_state_dict.items() if
        #                       key in self._model.state_dict()}

        # 加载匹配的权重到新模型
        self._sidechain.load_state_dict(adjusted_state_dict, strict=True)
        print('side chain load from state dict finished')

    def sample(self,sample_length):


        self.Proflow.set_device(self.device)
        X,C,S = self.Proflow.hybrid_Complex_sample(
            1, sample_length, self._model
        )

        p = Protein.from_XCS(X, C, S, )
        p.to_PDB(str(sample_length[0])+'_monomer.pdb')


    def sample_binder(self,Target=0,design_num=1):
        ref_pdb = '/home/junyu/project/binder_target/1nkp/nk/1nkp.pkl'

        model_name=self._exp_cfg.warm_start.split('/')[-3]


        design_name=ref_pdb.split('/')[-1].split('.')[0]
        design_path='/home/junyu/project/binder_target/1nkp//'
        ref_path='/home/junyu/project/binder_target/1nkp//reference.pkl'
        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path=colletdata()

        folder_path=design_path+model_name+'_monomer/'
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)
            print(f"文件夹'{folder_path}'已创建。")
        else:
            # 如果文件夹已存在
            print(f"文件夹'{folder_path}'已存在。")


        PdbDataset=sPdbDataset(ref_path, is_training=False)


        for ref_data in PdbDataset:
            # ref_data = du.read_pkl(ref_pdb)
            # 为每个Tensor增加一个空白维度
            del ref_data['csv_idx']
            ref_data = {
                key: torch.tensor(value).unsqueeze(0).to(self.device) 
                if isinstance(value, np.ndarray) else value 
                for key, value in ref_data.items()
            }

            com_idx=ref_data['com_idx']
            binder_mask=com_idx!=Target
            binder_mask=binder_mask.to(self.device)

            self.Proflow.set_device(self.device)
            noisy_batch = self.Proflow.corrupt_batch_binder(ref_data, 'fp32', binder_mask,t=0,design=True)



            samples = self.Proflow.hybrid_binder_sample(
                self._model,
                noisy_batch

             )
            X=samples[0]
            C = samples[1]
            S = samples[2]
            BF = samples[3]


            p = Protein.from_XCSB(X, C, S, BF)
            p.to_PDB(folder_path
            +design_name+'_'+str(design_num)+'_design.pdb')

    def sample_binder_withside(self,Target=0,design_num=1):
        # ref_pdb = '/home/junyu/project/binder_target/5o45/preprocessed/1ycr.pkl'

        model_name=self._exp_cfg.warm_start.split('/')[-3]


        design_name='5o45'
        design_path='/home/junyu/project/binder_target/5o45//'
        ref_path='/home/junyu/project/binder_target/5o45/preprocessed/reference.pkl'
        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path=colletdata()

        folder_path=design_path+model_name+'_sample_binder_withside/'
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)
            print(f"文件夹'{folder_path}'已创建。")
        else:
            # 如果文件夹已存在
            print(f"文件夹'{folder_path}'已存在。")


        PdbDataset=sPdbDataset(ref_path, is_training=False)


        for ref_data in PdbDataset:
            # ref_data = du.read_pkl(ref_pdb)
            # 为每个Tensor增加一个空白维度
            del ref_data['csv_idx']
            ref_data = {
                key: torch.tensor(value).unsqueeze(0).to(self.device) 
                if isinstance(value, np.ndarray) else value 
                for key, value in ref_data.items()
            }

            com_idx=ref_data['com_idx']
            binder_mask=com_idx!=Target
            binder_mask=binder_mask.to(self.device)

            self.Proflow.set_device(self.device)
            noisy_batch = self.Proflow.corrupt_batch_binder(ref_data, 'fp32', binder_mask,t=0,design=True)

            noisy_batch['ss']=noisy_batch['ss']*binder_mask
            noisy_batch['aatype'] = noisy_batch['aatype'] * binder_mask
            noisy_batch['chi'] = noisy_batch['chi'] * binder_mask[..., None]
            noisy_batch['atoms14_b_factors'] = noisy_batch['atoms14_b_factors'] * binder_mask[..., None]

            # sample str and seq and same time
            samples = self.Proflow.hybrid_binder_side_sample(
                self._model,
                noisy_batch,
                self._sidechain

             )
            X=samples[0]
            C = samples[1]
            S = samples[2]
            BF = samples[3]

            #design side


            p = Protein.from_XCSB(X, C, S, BF)
            p.to_PDB(folder_path
            +design_name+'_'+str(design_num)+'_design.pdb')
    def sample_binder_withside_randomlen(self,Target=0,design_num=1,min_length=30):
        # ref_pdb = '/home/junyu/project/binder_target/1nkp/nk/1nkp.pkl'

        model_name=self._exp_cfg.warm_start.split('/')[-3]



        design_path='/home/junyu/project/binder_target/1ycr/'
        ref_path='/home/junyu/project/binder_target/1ycr/preprocessed/reference.pkl'

        design_name=design_path.split('/')[-2]
        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path=colletdata()

        folder_path=design_path+model_name+'_random_min_seq_no_noise'+str(min_length)+'/'
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)
            print(f"文件夹'{folder_path}'已创建。")
        else:
            # 如果文件夹已存在
            print(f"文件夹'{folder_path}'已存在。")


        PdbDataset=sPdbDataset(ref_path, is_training=False)


        for ref_data in PdbDataset:
            # ref_data = du.read_pkl(ref_pdb)
            # 为每个Tensor增加一个空白维度
            del ref_data['csv_idx']

            # 获取com_idx并创建binder_mask
            com_idx=ref_data['com_idx']
            binder_mask=com_idx!=Target    # pos need to redesign
            binder_mask=binder_mask.to(self.device)

            # 获取目标长度和最小长度
            # 找到目标部分的起点
            target_start_idx = torch.where(binder_mask == False)[0].min().item()
            target_length = (1-binder_mask.cpu().numpy()).sum()


            # 随机采样一个长度
            sample_length = target_length#random.randint(min_length, target_length)

            # 随机选择连续的sample_length个位置，保证从目标起点开始
            start_idx = random.randint(target_start_idx, target_start_idx + target_length - sample_length)
            end_idx = start_idx + sample_length

            # 重新取连续的sample_length个位置，对于目标部分获取start_idx:end_idx之间的数据，非目标部分保留所有数据
            ref_data_resample = {}
            for key, value in ref_data.items():
                if value.shape[0] > target_start_idx:  # 确保 key 具有足够的长度
                    ref_data_resample[key] = torch.cat(
                        [value[ :target_start_idx, ...], value[ start_idx:end_idx, ...]], dim=0).unsqueeze(0).to(
                        self.device)
                else:
                    ref_data_resample[key] = value.unsqueeze(0).to(self.device)

            # re获取com_idx并创建binder_mask
            com_idx = ref_data_resample['com_idx']
            binder_mask = com_idx != Target
            binder_mask = binder_mask.to(self.device)

            self.Proflow.set_device(self.device)
            noisy_batch = self.Proflow.corrupt_batch_binder(ref_data_resample, 'fp32', binder_mask,t=0,design=True)

            noisy_batch['ss']=noisy_batch['ss']*binder_mask
            noisy_batch['aatype'] = noisy_batch['aatype'] * binder_mask
            noisy_batch['chi'] = noisy_batch['chi'] * binder_mask[..., None]
            noisy_batch['atoms14_b_factors'] = noisy_batch['atoms14_b_factors'] * binder_mask[..., None]

            # sample str and seq and same time
            samples = self.Proflow.hybrid_binder_side_sample(
                self._model,
                noisy_batch,
                self._sidechain

             )
            X=samples[0]
            C = samples[1]
            S = samples[2]
            BF = samples[3]

            #design side


            p = Protein.from_XCSB(X, C, S, BF)
            p.to_PDB(folder_path
            +design_name+'_'+str(design_num)+f'_{sample_length}_design.pdb')


    def sample_binder_withside_randomlen_motifsample(self,Target=0,design_num=1,min_length=30):
        # ref_pdb = '/home/junyu/project/binder_target/1nkp/nk/1nkp.pkl'

        model_name=self._exp_cfg.warm_start.split('/')[-3]



        design_path='/home/junyu/project/binder_target/1ycr/'
        ref_path='/home/junyu/project/binder_target/1ycr/preprocessed/reference.pkl'

        design_name=design_path.split('/')[-1]
        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path=colletdata()

        folder_path=design_path+model_name+'_random_min_seq_noise'+str(min_length)+'/'
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)
            print(f"文件夹'{folder_path}'已创建。")
        else:
            # 如果文件夹已存在
            print(f"文件夹'{folder_path}'已存在。")


        PdbDataset=sPdbDataset(ref_path, is_training=False)


        for ref_data in PdbDataset:
            # ref_data = du.read_pkl(ref_pdb)
            # 为每个Tensor增加一个空白维度
            del ref_data['csv_idx']

            # 获取com_idx并创建binder_mask
            com_idx=ref_data['com_idx']
            binder_mask=com_idx!=Target    # pos need to redesign
            binder_mask=binder_mask.to(self.device)

            # 获取目标长度和最小长度
            # 找到目标部分的起点
            target_start_idx = torch.where(binder_mask == False)[0].min().item()
            target_length = (1-binder_mask.cpu().numpy()).sum()


            # 随机采样一个长度
            sample_length = random.randint(min_length, target_length)

            # 随机选择连续的sample_length个位置，保证从目标起点开始
            start_idx = random.randint(target_start_idx, target_start_idx + target_length - sample_length)
            end_idx = start_idx + sample_length

            # 重新取连续的sample_length个位置，对于目标部分获取start_idx:end_idx之间的数据，非目标部分保留所有数据
            ref_data_resample = {}
            for key, value in ref_data.items():
                if value.shape[0] > target_start_idx:  # 确保 key 具有足够的长度
                    ref_data_resample[key] = torch.cat(
                        [value[ :target_start_idx, ...], value[ start_idx:end_idx, ...]], dim=0).unsqueeze(0).to(
                        self.device)
                else:
                    ref_data_resample[key] = value.unsqueeze(0).to(self.device)

            # re获取com_idx并创建binder_mask
            com_idx = ref_data_resample['com_idx']
            binder_mask = com_idx != Target
            binder_mask = binder_mask.to(self.device)

            self.Proflow.set_device(self.device)
            noisy_batch = self.Proflow.corrupt_batch_binder(ref_data_resample, 'fp32', binder_mask,t=0,design=True)

            noisy_batch['ss']=noisy_batch['ss']*binder_mask
            noisy_batch['aatype'] = noisy_batch['aatype'] * binder_mask
            noisy_batch['chi'] = noisy_batch['chi'] * binder_mask[..., None]
            noisy_batch['atoms14_b_factors'] = noisy_batch['atoms14_b_factors'] * binder_mask[..., None]

            # sample str and seq and same time
            samples = self.Proflow.hybrid_binder_side_sample(
                self._model,
                noisy_batch,
                self._sidechain

             )
            X=samples[0]
            C = samples[1]
            S = samples[2]
            BF = samples[3]

            #design side


            p = Protein.from_XCSB(X, C, S, BF)
            p.to_PDB(folder_path
            +design_name+'_'+str(design_num)+f'_{sample_length}_design.pdb')

    def sample_binder_bylength(self,Target=3,design_num=1,Length=100):
        """
        根据指定长度生成binder序列。

        Args:
            Target (int): 指定要重新设计的链的索引。这个参数标识哪条链需要被重新设计，
                         而不是靶点蛋白的链。例如，如果Target=3，那么com_idx=3的链会被重新设计，
                         而其他链（包括靶点蛋白）会保持不变。
            design_num (int): 要设计的序列数量
            Length (int): 期望的序列长度

        Returns:
            生成的binder序列和相关信息
        """
        model_name=self._inf_cfg.ckpt_path.split('/')[-3]

        design_class_name = '1bj1'
        design_path = '/home/junyu/project/binder_target/1bj1/preprocessed/'
        ref_path = '//home/junyu/project/binder_target/1bj1/preprocessed/reference.pkl'
        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path=colletdata()

        folder_path=design_path+model_name+'_bylen_pdbbind_'+'/'
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)
            print(f"文件夹'{folder_path}'已创建。")
        else:
            # 如果文件夹已存在
            print(f"文件夹'{folder_path}'已存在。")


        PdbDataset=sPdbDataset(ref_path, is_training=False)



        for i in range(len(PdbDataset)):#PdbDataset:
            design_name=design_class_name+'_'#+str(native[i])
            ref_data=PdbDataset[i]
            # ref_data = du.read_pkl(ref_pdb)
            # 为每个Tensor增加一个空白维度
            ref_data = {
                key: torch.tensor(value).unsqueeze(0).to(self.device)
                if isinstance(value, np.ndarray) else value
                for key, value in ref_data.items()
            }



            com_idx=ref_data['com_idx']
            Target=torch.tensor(Target,device=com_idx.device)
            fixed_mask = ~torch.isin(com_idx, Target)

            #make new com_idx
            desgin_comidx=Target[0]
            binder_com_idx=torch.ones(size=(com_idx.shape[0],Length),device=self.device)*desgin_comidx


            new_com_idx=torch.cat([com_idx[fixed_mask][None,:],binder_com_idx],dim=-1)

            if (~fixed_mask).sum()==0:
                target_chain=2
            else:
                target_chain=list(set(ref_data['chain_idx'][~fixed_mask].tolist()))[0]

            new_batch={
                'atoms14':torch.cat([ref_data['atoms14'][fixed_mask][None,:],torch.rand(size=(ref_data['atoms14'].shape[0],Length,ref_data['atoms14'].shape[2],ref_data['atoms14'].shape[3]),device=self.device)],dim=1),
                'atoms14_b_factors': torch.cat([ref_data['atoms14_b_factors'][fixed_mask][None,:], torch.rand(size=(
                ref_data['atoms14_b_factors'].shape[0], Length, ref_data['atoms14_b_factors'].shape[2]),device=self.device)],
                                     dim=1),
                'chi': torch.cat([ref_data['chi'][fixed_mask][None,:], torch.ones(size=(
                    ref_data['chi'].shape[0], Length, ref_data['chi'].shape[2]),device=self.device)],
                                               dim=1),

                'mask_chi': torch.cat([ref_data['mask_chi'][fixed_mask][None,:], torch.zeros(size=(
                    ref_data['mask_chi'].shape[0], Length, ref_data['mask_chi'].shape[2]),device=self.device)],
                                 dim=1),

                'chain_idx': torch.cat([ref_data['chain_idx'][fixed_mask][None,:], target_chain* torch.ones(size=(
                    ref_data['chain_idx'].shape[0], Length),device=self.device)],
                                 dim=1),

                'aatype': torch.cat([ref_data['aatype'][fixed_mask][None,:], 20 * torch.ones(size=(
                    ref_data['aatype'].shape[0], Length),device=self.device,dtype=torch.long)],
                                       dim=1),

                'ss': torch.cat([ref_data['ss'][fixed_mask][None,:],  torch.zeros(size=(
                    ref_data['ss'].shape[0], Length),device=self.device,dtype=torch.long)],
                                    dim=1),

                'res_idx': torch.cat([ref_data['res_idx'][fixed_mask][None,:], torch.range(1, Length).unsqueeze(0).to(self.device)],
                                dim=1),

                'res_mask': torch.cat([ref_data['res_mask'][fixed_mask][None,:],  torch.ones(size=(
                    ref_data['ss'].shape[0], Length),device=self.device)],
                                    dim=1),

                'com_idx': new_com_idx

            }

            fixed_mask = ~torch.isin(new_com_idx, desgin_comidx)
            fixed_mask = fixed_mask.to(self.device)

            # 设置设备
            self.Proflow.set_device(self.device)
            new_batch['fixed_mask'] = fixed_mask

            # 使用Proflow进行采样
            noisy_batch = self.Proflow.corrupt_batch_binder(
                new_batch, 'fp32', fixed_mask, design_num, t=0, design=True, path=design_path
            )

            # 生成样本
            samples = self.Proflow.hybrid_binder_sample(
                self._model,
                noisy_batch,
                num_steps=500,
            )

            # 解包样本
            X, C, S, BF = samples

            # 保存结果
            p = Protein.from_XCSB(X, C, S, BF)
            p.to_PDB(folder_path + design_name + '_' + str(design_num) + '_design.pdb')



    def sample_binder_bylength_reference(self, Target=0, design_num=1, Length=30, fixed_com_idx=[], hotspot=None, 
                                        design_class_name=None, design_path=None, ref_path=None):
        '''
        根据参考结构生成binder序列。

        Args:
            Target (int): 设计目标的com_idx
            design_num (int): 设计序列的数量
            Length (int): 期望的序列长度
            fixed_com_idx (list): 固定的com_idx列表
            hotspot (list, optional): 热点残基列表
            design_class_name (str): 设计类名称
            design_path (str): 设计输出路径
            ref_path (str): 参考文件路径
        '''
        # 从配置中获取参数，如果没有提供则使用默认值
        if design_class_name is None:
            design_class_name = self._inf_cfg.design_class_name
        if design_path is None:
            design_path = os.path.join(self._inf_cfg.base_path, 'design/')
        if ref_path is None:
            ref_path = self._inf_cfg.ref_path

        model_name = self._inf_cfg.ckpt_path.split('/')[-2]

        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path = colletdata()

        folder_path = design_path + model_name + '_byreference_' + '/'
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)
            print(f"文件夹'{folder_path}'已创建。")
        else:
            # 如果文件夹已存在
            print(f"文件夹'{folder_path}'已存在。")

        PdbDataset = sPdbDataset(ref_path, is_training=False)

        for i in range(len(PdbDataset)):
            design_name = design_class_name
            ref_data = PdbDataset[i]
            # 为每个Tensor增加一个空白维度
            ref_data = {
                key: torch.tensor(value).unsqueeze(0).to(self.device) 
                if isinstance(value, np.ndarray) else value.unsqueeze(0).to(self.device)
                for key, value in ref_data.items()
            }

            design_index, _ = calculate_interface_residues_vtarget(ref_data['atoms14'][0], ref_data['com_idx'][0], 
                                                                 fixed_com_idx=fixed_com_idx, n=Length)

            com_idx = ref_data['com_idx']
            Target = torch.tensor(Target, device=com_idx.device)
            binder_mask = ~torch.isin(com_idx, Target)  # this is fixed

            # make new com_idx
            desgin_comidx = Target[0]
            binder_com_idx = torch.ones(size=(com_idx.shape[0], Length), device=self.device) * desgin_comidx

            new_com_idx = torch.cat([com_idx[binder_mask][None, :], binder_com_idx], dim=-1)
            target_chain = list(set(ref_data['chain_idx'][~binder_mask].tolist()))[0]

            new_batch = {
                'atoms14': torch.cat((ref_data['atoms14'][binder_mask][None, :], ref_data['atoms14'][0][design_index].unsqueeze(0)), dim=1),
                'atoms14_b_factors': torch.cat([ref_data['atoms14_b_factors'][binder_mask][None, :], ref_data['atoms14_b_factors'][0][design_index].unsqueeze(0)],
                                     dim=1),
                'chi': torch.cat([ref_data['chi'][binder_mask][None, :], ref_data['chi'][0][design_index].unsqueeze(0)],
                               dim=1),
                'mask_chi': torch.cat([ref_data['mask_chi'][binder_mask][None, :], ref_data['mask_chi'][0][design_index].unsqueeze(0)],
                                 dim=1),
                'chain_idx': torch.cat([ref_data['chain_idx'][binder_mask][None, :], target_chain * torch.ones(size=(
                    ref_data['chain_idx'].shape[0], Length), device=self.device)],
                                 dim=1),
                'aatype': torch.cat([ref_data['aatype'][binder_mask][None, :], 20 * torch.ones(size=(
                    ref_data['aatype'].shape[0], Length), device=self.device, dtype=torch.long)],
                               dim=1),
                'ss': torch.cat([ref_data['ss'][binder_mask][None, :], torch.zeros(size=(
                    ref_data['ss'].shape[0], Length), device=self.device, dtype=torch.long)],
                            dim=1),
                'res_idx': torch.cat([ref_data['res_idx'][binder_mask][None, :], torch.range(1, Length).unsqueeze(0).to(self.device)],
                                dim=1),
                'res_mask': torch.cat([ref_data['res_mask'][binder_mask][None, :], torch.ones(size=(
                    ref_data['ss'].shape[0], Length), device=self.device)],
                                    dim=1),
                'com_idx': new_com_idx
            }

            binder_mask = ~torch.isin(new_com_idx, desgin_comidx)
            binder_mask = binder_mask.to(self.device)

            self.Proflow.set_device(self.device)
            new_batch['fixed_mask'] = binder_mask

            if hotspot:
                center_of_mass, centered_atoms14 = self.compute_center_of_mass(hotspot, new_batch['chain_idx'][0],
                                                                             new_batch['res_idx'][0],
                                                                             new_batch['atoms14'][0])
                new_batch['atoms14'] = centered_atoms14.unsqueeze(0)
                noisy_batch = self.Proflow.corrupt_batch_binder(new_batch, 'fp32', binder_mask, design_num, t=0,
                                                              design=False, path=design_path, hotspot=hotspot)
            else:
                noisy_batch = self.Proflow.corrupt_batch_binder(new_batch, 'fp32', binder_mask, design_num, t=0, 
                                                              design=False, path=design_path)

            samples = self.Proflow.hybrid_binder_sample(
                self._model,
                noisy_batch,
                num_steps=1000,
            )
            
            X = samples[0]
            C = samples[1]
            S = samples[2]
            BF = samples[3]

            p = Protein.from_XCSB(X, C, S, BF)
            p.to_PDB(folder_path + design_name + '_' + str(design_num) + '_design.pdb')



    def sample_binder_reference_chains(self, target_com_idx=1, reference_chains=[2], design_num=1, Length=100):
        """
        根据指定的目标com_idx和参考链生成binder序列。这个方法会保留指定的target_com_idx和reference_chains对应的部分，
        其余部分会被删除，然后添加新的待设计的binder部分。

        Args:
            target_com_idx (int): 指定靶点所在的com_idx，这部分是不会被修改的
            reference_chains (list): 要保留的参考链的chain_idx列表
            design_num (int): 要设计的序列数量
            Length (int): 期望的序列长度

        Returns:
            生成的binder序列和相关信息，结果会保存为PDB文件
        """
        model_name = self._inf_cfg.ckpt_path.split('/')[-3]
        design_class_name = '1bj1'
        design_path = '/home/junyu/project/binder_target/1bj1/preprocessed/'
        ref_path = '//home/junyu/project/binder_target/1bj1/preprocessed/reference.pkl'

        # 检查并创建文件
        if not os.path.exists(ref_path):
            from data.collect_pkl import colletdata
            ref_path = colletdata()

        # 创建输出文件夹
        folder_path = design_path + model_name + '_bylen_pdbbind_' + '/'
        os.makedirs(folder_path, exist_ok=True)

        PdbDataset = sPdbDataset(ref_path, is_training=False)

        for i in range(len(PdbDataset)):
            design_name = design_class_name + '_'
            ref_data = PdbDataset[i]
            
            # 转换数据到tensor
            ref_data = {
                key: torch.tensor(value).unsqueeze(0).to(self.device) 
                if isinstance(value, np.ndarray) else value.unsqueeze(0).to(self.device)
                for key, value in ref_data.items()
            }

            # 创建保留mask：保留目标com_idx和参考链的部分
            com_mask = (ref_data['com_idx'] == target_com_idx)
            # chain_mask = torch.zeros_like(ref_data['chain_idx'], dtype=torch.bool)
            # for chain in reference_chains:
            #     chain_mask = chain_mask | (ref_data['chain_idx'] == chain)
            
            # 合并mask：保留目标com_idx或参考链的部分
            keep_mask = com_mask#(com_mask | chain_mask)  # 移除batch维度

            del ref_data['csv_idx']
            # 过滤数据，只保留需要的部分
            # filtered_data = {}
            # for key, value in ref_data.items():
            #         filtered_data[key] = value[keep_mask]

            # from data.interpolant import save_pdb_chain
            # bb = filtered_data['atoms14'][..., :4, :].cpu().numpy()
            # chain_idx = filtered_data['chain_idx'].cpu().numpy()
            # save_pdb_chain(bb.reshape(-1, 3), chain_idx, 'test_1bj1.pdb')

            # 创建新的binder部分
            binder_com_idx =  1
            binder_chain_idx = reference_chains[0]

            binder_com_idx_pad=torch.ones(size=(Length,),device=ref_data['com_idx'].device)*binder_com_idx

            fixed_mask = keep_mask

            new_com_idx=torch.cat([ref_data['com_idx'][fixed_mask][None,:],binder_com_idx_pad],dim=-1).unsqueeze(0)


            # 合并原始数据和新的binder部分
            new_batch = {
                'atoms14': torch.cat([ref_data['atoms14'][fixed_mask][None, :], torch.rand(size=(
                ref_data['atoms14'].shape[0], Length, ref_data['atoms14'].shape[2], ref_data['atoms14'].shape[3]),
                                                                                           device=self.device)], dim=1),
                'atoms14_b_factors': torch.cat([ref_data['atoms14_b_factors'][fixed_mask][None, :], torch.rand(size=(
                    ref_data['atoms14_b_factors'].shape[0], Length, ref_data['atoms14_b_factors'].shape[2]),
                    device=self.device)],
                                               dim=1),
                'chi': torch.cat([ref_data['chi'][fixed_mask][None, :], torch.ones(size=(
                    ref_data['chi'].shape[0], Length, ref_data['chi'].shape[2]), device=self.device)],
                                 dim=1),

                'mask_chi': torch.cat([ref_data['mask_chi'][fixed_mask][None, :], torch.zeros(size=(
                    ref_data['mask_chi'].shape[0], Length, ref_data['mask_chi'].shape[2]), device=self.device)],
                                      dim=1),

                'chain_idx': torch.cat([ref_data['chain_idx'][fixed_mask][None, :], binder_chain_idx * torch.ones(size=(
                    ref_data['chain_idx'].shape[0], Length), device=self.device)],
                                       dim=1),

                'aatype': torch.cat([ref_data['aatype'][fixed_mask][None, :], 20 * torch.ones(size=(
                    ref_data['aatype'].shape[0], Length), device=self.device, dtype=torch.long)],
                                    dim=1),

                'ss': torch.cat([ref_data['ss'][fixed_mask][None, :], torch.zeros(size=(
                    ref_data['ss'].shape[0], Length), device=self.device, dtype=torch.long)],
                                dim=1),

                'res_idx': torch.cat(
                    [ref_data['res_idx'][fixed_mask][None, :], torch.range(1, Length).unsqueeze(0).to(self.device)],
                    dim=1),

                'res_mask': torch.cat([ref_data['res_mask'][fixed_mask][None, :], torch.ones(size=(
                    ref_data['ss'].shape[0], Length), device=self.device)],
                                      dim=1),

                'com_idx': new_com_idx

            }

            fixed_mask = ~torch.isin(new_com_idx, binder_chain_idx)
            fixed_mask = fixed_mask.to(self.device)

            # 设置设备
            self.Proflow.set_device(self.device)
            new_batch['fixed_mask'] = fixed_mask

            # 使用Proflow进行采样
            noisy_batch = self.Proflow.corrupt_batch_binder(
                new_batch, 'fp32', fixed_mask, design_num, t=0, design=True, path=design_path
            )

            # 生成样本
            samples = self.Proflow.hybrid_binder_sample(
                self._model,
                noisy_batch,
                num_steps=500,
            )

            # 解包样本
            X, C, S, BF = samples

            # 保存结果
            p = Protein.from_XCSB(X, C, S, BF)
            p.to_PDB(folder_path + design_name + '_' + str(design_num) + '_design.pdb')



    def sample_binder_bylength_hotspot(self,Target=[1],design_num=1,Length=30,hotspot=[],fixed_by_chain=None,design_class_name = 'gpcr',        base_path =None,        ref_path = None):
        '''
        根据指定的目标和热点残基生成binder序列。

        Args:
            Target (list): 设计目标的com_idx列表，binder  指定需要被重新设计的复合物索引,   如果原来的PDB 只有一条靶点链条，则随便指定一个与靶点不重复的com值
            design_num (int): 设计序列的数量
            Length (int): 期望的序列长度
            hotspot (list): 热点残基列表，用于指定重要的相互作用位点
            fixed_by_chain (list, optional): 用于进一步控制设计目标。当一个com_idx包含多条链时，
                                           可以通过此参数指定只针对特定的链进行设计。
                                           例如：com_idx=1包含chain_idx=[1,2,3]，
                                           如果fixed_by_chain=[2]，则只会对chain_idx=2的部分进行设计(as target)，
                                           其他链即使在同一个com_idx中也会保持不变。
                                           默认为None，表示对指定com_idx的所有链都进行设计。

        Returns:
            生成的binder序列和相关信息，结果会保存为PDB文件
        '''
        model_name=self._inf_cfg.ckpt_path.split('/')[-2]



        
        # Determine folder based on parameters
        folder_suffix = '_'
        if len(hotspot) > 0:
            folder_suffix += f'hotspot_{"".join(map(str, hotspot))}'
        else:
            folder_suffix += '_no_hotspot'
            
        if fixed_by_chain is not None:
            folder_suffix += f'_fixed_chain_{",".join(map(str, fixed_by_chain))}'
        else:
            folder_suffix += '_only_com_idx'


        folder_suffix=folder_suffix+'_'+str(self.sample_step)+'s'
        design_path = os.path.join(base_path, folder_suffix)
        if not os.path.exists(design_path):
            os.makedirs(design_path)
            


        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path=colletdata()

        folder_path = os.path.join(design_path, f'{model_name}_/')
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)
            


        PdbDataset = sPdbDataset(ref_path, is_training=False)

        for i in range(len(PdbDataset)):#PdbDataset:
            design_name=design_class_name+'_'#+str(native[i])
            ref_data=PdbDataset[i]
            # ref_data = du.read_pkl(ref_pdb)
            # 为每个Tensor增加一个空白维度
            ref_data = {
                key: torch.tensor(value).unsqueeze(0).to(self.device) 
                if isinstance(value, np.ndarray) else value.unsqueeze(0).to(self.device)
                for key, value in ref_data.items()
            }



            com_idx=ref_data['com_idx']
            Target=torch.tensor(Target,device=com_idx.device)
            fixed_mask = ~torch.isin(com_idx, Target)

            if fixed_by_chain is not None:
                chain_target = torch.tensor(fixed_by_chain, device=com_idx.device)
                fixed_chain=torch.isin(ref_data['chain_idx'], chain_target)

                fixed_mask=fixed_mask & fixed_chain

            #make new com_idx
            desgin_comidx=Target[0]
            binder_com_idx=torch.ones(size=(com_idx.shape[0],Length),device=com_idx[fixed_mask].device)*desgin_comidx



            new_com_idx=torch.cat([com_idx[fixed_mask][None,:],binder_com_idx],dim=-1)



            if (~fixed_mask).sum()==0:
                target_chain=2
            else:
                target_chain=list(set(ref_data['chain_idx'][~fixed_mask].tolist()))[0]

            new_batch={
                'atoms14':torch.cat([ref_data['atoms14'][fixed_mask][None,:],torch.rand(size=(ref_data['atoms14'].shape[0],Length,ref_data['atoms14'].shape[2],ref_data['atoms14'].shape[3]),device=self.device)],dim=1),
                'atoms14_b_factors': torch.cat([ref_data['atoms14_b_factors'][fixed_mask][None,:], torch.rand(size=(
                ref_data['atoms14_b_factors'].shape[0], Length, ref_data['atoms14_b_factors'].shape[2]),device=self.device)],
                                     dim=1),
                'chi': torch.cat([ref_data['chi'][fixed_mask][None,:], torch.ones(size=(
                    ref_data['chi'].shape[0], Length, ref_data['chi'].shape[2]),device=self.device)],
                                               dim=1),

                'mask_chi': torch.cat([ref_data['mask_chi'][fixed_mask][None,:], torch.zeros(size=(
                    ref_data['mask_chi'].shape[0], Length, ref_data['mask_chi'].shape[2]),device=self.device)],
                                 dim=1),

                'chain_idx': torch.cat([ref_data['chain_idx'][fixed_mask][None,:], target_chain* torch.ones(size=(
                    ref_data['chain_idx'].shape[0], Length),device=self.device)],
                                 dim=1),

                'aatype': torch.cat([ref_data['aatype'][fixed_mask][None,:], 20 * torch.ones(size=(
                    ref_data['aatype'].shape[0], Length),device=self.device,dtype=torch.long)],
                                       dim=1),

                'ss': torch.cat([ref_data['ss'][fixed_mask][None,:],  torch.zeros(size=(
                    ref_data['ss'].shape[0], Length),device=self.device,dtype=torch.long)],
                                    dim=1),

                'res_idx': torch.cat([ref_data['res_idx'][fixed_mask][None,:], torch.range(1, Length).unsqueeze(0).to(self.device)],
                                dim=1),

                'res_mask': torch.cat([ref_data['res_mask'][fixed_mask][None,:],  torch.ones(size=(
                    ref_data['ss'].shape[0], Length),device=self.device)],
                                    dim=1),

                'com_idx': new_com_idx

            }

            fixed_mask= ~torch.isin(new_com_idx, desgin_comidx)
            fixed_mask=fixed_mask.to(self.device)

            self.Proflow.set_device(self.device)
            new_batch['fixed_mask']=fixed_mask

            if hotspot:
                center_of_mass, centered_atoms14=self.compute_center_of_mass(hotspot, new_batch['chain_idx'][0], new_batch['res_idx'][0], new_batch['atoms14'][0])
                new_batch['atoms14']=centered_atoms14.unsqueeze(0)
                noisy_batch = self.Proflow.corrupt_batch_binder(new_batch, '16-mixed', fixed_mask,design_num,t=0,design=True ,path=design_path ,hotspot=True )
            else:

                noisy_batch = self.Proflow.corrupt_batch_binder(new_batch, 'fp32', fixed_mask,design_num,t=0,design=True ,path=design_path ,hotspot=False )


            samples = self.Proflow.hybrid_binder_sample(
                self._model,
                noisy_batch,
                num_steps=self.sample_step,


             )
            X=samples[0]
            C = samples[1]
            S = samples[2]
            BF = samples[3]


            p = Protein.from_XCSB(X, C, S, BF)
            p.to_PDB(folder_path
            +design_name+'_'+str(design_num)+'_design.pdb')
        return folder_path

    def sample_DYNbinder_bylength_hotspot(self, Target=[1], design_num=1, Length=30, hotspot=[], fixed_by_chain=None,
                                       design_class_name='gpcr',
                                       base_path='/home/junyu/project/binder_target/1bj1/preprocessed',
                                       ref_path='/home/junyu/project/binder_target/1bj1/preprocessed/reference.pkl',
                                          add_path='/home/junyu/project/binder_target/1bj1/preprocessed/reference.pkl',
                                          add_target=None,
                                          add_fixed_by_chain=None):
        '''
        根据指定的目标和热点残基生成binder序列。

        Args:
            Target (list): 设计目标的com_idx列表，指定需要被重新设计的复合物索引
            design_num (int): 设计序列的数量
            Length (int): 期望的序列长度
            hotspot (list): 热点残基列表，用于指定重要的相互作用位点
            fixed_by_chain (list, optional): 用于进一步控制设计目标。当一个com_idx包含多条链时，
                                           可以通过此参数指定只针对特定的链进行设计。
                                           例如：com_idx=1包含chain_idx=[1,2,3]，
                                           如果fixed_by_chain=[2]，则只会对chain_idx=2的部分进行设计，
                                           其他链即使在同一个com_idx中也会保持不变。
                                           默认为None，表示对指定com_idx的所有链都进行设计。

        Returns:
            生成的binder序列和相关信息，结果会保存为PDB文件
        '''
        model_name = self._inf_cfg.ckpt_path.split('/')[-2]

        # Determine folder based on parameters
        folder_suffix = '_'
        if len(hotspot) > 0:
            folder_suffix += '_with_addhotspot'
        else:
            folder_suffix += '_no_hotspot'

        if fixed_by_chain is not None:
            folder_suffix += f'_fixed_chain_{",".join(map(str, fixed_by_chain))}'
        else:
            folder_suffix += '_only_com_idx'

        design_path = os.path.join(base_path, folder_suffix)
        if not os.path.exists(design_path):
            os.makedirs(design_path)

        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path = colletdata()

        folder_path = os.path.join(design_path, f'{model_name}_by_sample_binder_bylength_hotspot_from90/')
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)

        PdbDataset = sPdbDataset(ref_path, is_training=False)

        for i in range(len(PdbDataset)):  # PdbDataset:
            design_name = design_class_name + '_'  # +str(native[i])
            ref_data = PdbDataset[i]
            # ref_data = du.read_pkl(ref_pdb)
            # 为每个Tensor增加一个空白维度
            ref_data = {
                key: torch.tensor(value).unsqueeze(0).to(self.device)
                if isinstance(value, np.ndarray) else value.unsqueeze(0).to(self.device)
                for key, value in ref_data.items()
            }

            com_idx = ref_data['com_idx']
            Target = torch.tensor(Target, device=com_idx.device)
            fixed_mask = ~torch.isin(com_idx, Target)

            if fixed_by_chain is not None:
                chain_target = torch.tensor(fixed_by_chain, device=com_idx.device)
                fixed_chain = torch.isin(ref_data['chain_idx'], chain_target)

                fixed_mask = fixed_mask & fixed_chain

            # make new com_idx
            desgin_comidx = Target[0]
            binder_com_idx = torch.ones(size=(com_idx.shape[0], Length),
                                        device=com_idx[fixed_mask].device) * desgin_comidx

            new_com_idx = torch.cat([com_idx[fixed_mask][None, :], binder_com_idx], dim=-1)

            if (~fixed_mask).sum() == 0:
                target_chain = 2
            else:
                target_chain = list(set(ref_data['chain_idx'][~fixed_mask].tolist()))[0]

            new_batch = {
                'atoms14': torch.cat([ref_data['atoms14'][fixed_mask][None, :], torch.rand(size=(
                ref_data['atoms14'].shape[0], Length, ref_data['atoms14'].shape[2], ref_data['atoms14'].shape[3]),
                                                                                           device=self.device)], dim=1),
                'atoms14_b_factors': torch.cat([ref_data['atoms14_b_factors'][fixed_mask][None, :], torch.rand(size=(
                    ref_data['atoms14_b_factors'].shape[0], Length, ref_data['atoms14_b_factors'].shape[2]),
                    device=self.device)],
                                               dim=1),
                'chi': torch.cat([ref_data['chi'][fixed_mask][None, :], torch.ones(size=(
                    ref_data['chi'].shape[0], Length, ref_data['chi'].shape[2]), device=self.device)],
                                 dim=1),

                'mask_chi': torch.cat([ref_data['mask_chi'][fixed_mask][None, :], torch.zeros(size=(
                    ref_data['mask_chi'].shape[0], Length, ref_data['mask_chi'].shape[2]), device=self.device)],
                                      dim=1),

                'chain_idx': torch.cat([ref_data['chain_idx'][fixed_mask][None, :], target_chain * torch.ones(size=(
                    ref_data['chain_idx'].shape[0], Length), device=self.device)],
                                       dim=1),

                'aatype': torch.cat([ref_data['aatype'][fixed_mask][None, :], 20 * torch.ones(size=(
                    ref_data['aatype'].shape[0], Length), device=self.device, dtype=torch.long)],
                                    dim=1),

                'ss': torch.cat([ref_data['ss'][fixed_mask][None, :], torch.zeros(size=(
                    ref_data['ss'].shape[0], Length), device=self.device, dtype=torch.long)],
                                dim=1),

                'res_idx': torch.cat(
                    [ref_data['res_idx'][fixed_mask][None, :], torch.range(1, Length).unsqueeze(0).to(self.device)],
                    dim=1),

                'res_mask': torch.cat([ref_data['res_mask'][fixed_mask][None, :], torch.ones(size=(
                    ref_data['ss'].shape[0], Length), device=self.device)],
                                      dim=1),

                'com_idx': new_com_idx

            }

            fixed_mask = ~torch.isin(new_com_idx, desgin_comidx)
            fixed_mask = fixed_mask.to(self.device)


            self.Proflow.set_device(self.device)
            new_batch['fixed_mask'] = fixed_mask

            for i,add in enumerate( add_path):
                add_PdbDataset = sPdbDataset(add, is_training=False)
                add_data = add_PdbDataset[0]
                add_data = {
                    key: torch.tensor(value).unsqueeze(0).to(self.device)
                    if isinstance(value, np.ndarray) else value.unsqueeze(0).to(self.device)
                    for key, value in add_data.items()
                }
                # 修改：Target和fixed_by_chain分别从add_target和add_fixed_by_chain取
                add_com_idx = add_data['com_idx']
                add_Target = add_target[i]
                add_Target = torch.tensor(add_Target, device=com_idx.device)
                add_fixed_mask = ~torch.isin(add_com_idx, add_Target)

                add_fixed_by_chain = add_fixed_by_chain[i] if add_fixed_by_chain is not None else None
                if add_fixed_by_chain is not None:
                    add_chain_target = torch.tensor(add_fixed_by_chain, device=add_com_idx.device)
                    add_fixed_chain = torch.isin(add_data['chain_idx'], add_chain_target)
                    add_fixed_mask = add_fixed_mask & add_fixed_chain
                    print(f'add_fixed_mask {i}: ', add_fixed_mask.sum())
                    add_this_data=add_data['aatype'][add_fixed_mask][None, :]




                    # 1. 取 N = fixed_mask.sum()
                    N = fixed_mask.sum().item()
                    # 2. 取 add_data 里前 N 个
                    add_atoms14 = add_data['atoms14'][:, :fixed_mask.shape[-1], ...]*fixed_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 14, 3]
                    # 3. 取 new_batch 里 fixed_mask 前 N 个
                    new_atoms14 = new_batch['atoms14'][:, :fixed_mask.shape[-1],
                                  ...]  # [B, N, 14, 3]

                    # test1=new_atoms14-new_atoms14[...,1,:].unsqueeze(-2)
                    # test2 = add_atoms14 - add_atoms14[..., 1, :].unsqueeze(-2)
                    #
                    # aligned_add_atoms14=align_pred_to_true(add_atoms14[...,:150,:,:].reshape(1,-1,3),new_atoms14[...,:150,:,:].reshape(1,-1,3),fixed_mask.unsqueeze(-1).expand(-1, -1, 14)[...,:150,:].reshape(1,-1))
                    # rmsd = torch.sqrt(((aligned_add_atoms14 - new_atoms14) ** 2).sum() / (N*14))

                    # def rigid_align_by_subset(A: torch.Tensor, B: torch.Tensor, core_len=150):
                    #     """
                    #     Align structure A to structure B using only the first `core_len` residues.
                    #     A, B: [N, 3]
                    #     Return: A_aligned [N, 3], rotation matrix R, RMSD on core region
                    #     """
                    #     assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3
                    #     A_core = A[:core_len]
                    #     B_core = B[:core_len]
                    #
                    #     # 去质心
                    #     A_mean = A_core.mean(dim=0, keepdim=True)
                    #     B_mean = B_core.mean(dim=0, keepdim=True)
                    #     A_centered = A_core - A_mean
                    #     B_centered = B_core - B_mean
                    #
                    #     # Kabsch 旋转
                    #     U, _, Vt = torch.linalg.svd(A_centered.T @ B_centered)
                    #     R = Vt.T @ U.T
                    #     if torch.det(R) < 0:
                    #         Vt[-1] *= -1
                    #         R = Vt.T @ U.T
                    #
                    #     # 应用于整个结构 A
                    #     A_all_centered = A_centered - A_mean
                    #     A_aligned = A_all_centered @ R + B_mean  # 平移回 B 的中心
                    #
                    #     # 核心区 RMSD
                    #     rmsd = torch.sqrt(((A_aligned[:core_len] - B_centered) ** 2).sum() / core_len)
                    #
                    #     return A_aligned, B, rmsd
                    #
                    # aligned_add_atoms14, aligned_new_atoms14, rmsd=rigid_align_by_subset(add_atoms14.reshape(1, -1, 3).squeeze(0)    ,
                    #                                                                       new_atoms14.reshape(1, -1, 3).squeeze(0),
                    #                                                                       150*14)

                    # 4. 对齐
                    aligned_add_atoms14, aligned_new_atoms14, R = du.batch_align_structures(
                        add_atoms14.reshape(1, -1, 3)    , new_atoms14.reshape(1, -1, 3), mask=fixed_mask.unsqueeze(-1).expand(-1, -1, 14).reshape(1, -1)  # [1, N, 42]
                    )
                    # 5. 替换 add_data 里的前 N 个
                    add_data['atoms14']= aligned_add_atoms14.reshape(1,new_atoms14.shape[1],new_atoms14.shape[2] ,3)


                new_batch['atoms14']=aligned_new_atoms14.reshape(1,new_atoms14.shape[1],new_atoms14.shape[2] ,3)

                # 2. 将 add_data 的每一项添加进 new_batch，形成 batch size=2
                L = fixed_mask.shape[1]  # 以 fixed_mask 的长度为准

                for key in new_batch:
                    if key in add_data and isinstance(new_batch[key], torch.Tensor) and isinstance(add_data[key],
                                                                                                   torch.Tensor):
                        # 1. 先调整 add_data[key] 的长度为 L
                        add_val = add_data[key]
                        add_len = add_val.shape[1]
                        if add_len < L:
                            # padding
                            pad_shape = list(add_val.shape)
                            pad_shape[1] = L - add_len
                            pad_tensor = torch.zeros(pad_shape, dtype=add_val.dtype, device=add_val.device)
                            add_val = torch.cat([add_val, pad_tensor], dim=1)
                        elif add_len > L:
                            # 裁剪
                            add_val = add_val[:, :L, ...]
                        # 2. mask
                        mask = fixed_mask
                        while mask.dim() < add_val.dim():
                            mask = mask.unsqueeze(-1)
                        masked_add_val = add_val * mask
                        # 3. 拼接
                        new_batch[key] = torch.cat([new_batch[key], masked_add_val], dim=0)
                B = fixed_mask.shape[0]  # 以 fixed_mask 的长度为准

                new_batch['fixed_mask'] = fixed_mask.expand(B+len(add_path),-1)
            if hotspot:
                center_of_mass, centered_atoms14 = self.compute_center_of_mass(hotspot, new_batch['chain_idx'][0],
                                                                               new_batch['res_idx'][0],
                                                                               new_batch['atoms14'])



                new_batch['atoms14'] = centered_atoms14

                t1=new_batch['atoms14'][0]
                t2 = new_batch['atoms14'][1]
                t0=t2-t1
                noisy_batch = self.Proflow.corrupt_batch_DYNbinder(new_batch, 'fp32',  new_batch['fixed_mask'] , design_num, t=0,
                                                                design=True, path=design_path, hotspot=True)
            else:

                noisy_batch = self.Proflow.corrupt_batch_DYNbinder(new_batch, 'fp32', fixed_mask, design_num, t=0,
                                                                design=True, path=design_path, hotspot=False)

            samples = self.Proflow.hybrid_DYN_binder_sample(
                self._model,
                noisy_batch,
                num_steps=200,

            )
            for i in range(samples[0].shape[0]):

                    X = samples[0][i].unsqueeze(0)
                    C = samples[1][0].unsqueeze(0)
                    S = samples[2][i].unsqueeze(0)
                    BF = samples[3][i].unsqueeze(0)

                    p = Protein.from_XCSB(X, C, S, BF)
                    p.to_PDB(folder_path
                             + design_name + '_' + str(design_num) + '_design_batch' + str(i) + '.pdb')

    def parse_hotspot(self,hotspot, chain_idx):
        """
        解析hotspot列表，返回对应的链索引和残基索引的元组列表
        """
        # 获取唯一的链索引
        unique_chain_indices = sorted(set(chain_idx.cpu().numpy().tolist()))

        # 构建链字母到实际索引的映射，只映射实际存在的链索引
        # 构建链字母到实际索引的映射
        chain_map = {chr(ord('A') + int(idx) - 1): idx for idx in unique_chain_indices}

        parsed_hotspots = []
        for hs in hotspot:
            chain = hs[0]
            res = int(hs[1:])
            parsed_hotspots.append((chain_map[chain], res))
        return parsed_hotspots

    def compute_center_of_mass(self,hotspot, chain_idx, res_idx, atoms14):
        """
        计算指定hotspot位置的Ca原子的重心，并用其中心化所有原子坐标
        """
        parsed_hotspots = self.parse_hotspot(hotspot, chain_idx)

        selected_atoms = []
        for chain, res in parsed_hotspots:
            mask = (chain_idx == chain) & (res_idx == res)

            if len(atoms14.shape)==4:

                selected_atoms.append(atoms14[:,mask, 1, :])  # 提取Ca原子的坐标
            else:
                selected_atoms.append(atoms14[mask, 1, :])

        if len(selected_atoms) == 0:
            raise ValueError("No atoms found for the given hotspots.")

        # 合并所有选定的原子
        selected_atoms = torch.cat(selected_atoms, dim=len(atoms14.shape)-3)

        # 计算重心
        center_of_mass = selected_atoms.mean(dim=len(atoms14.shape)-3)

        # 使用重心中心化所有的原子坐标
        if len(atoms14.shape) == 4:
            centered_atoms14 = atoms14 - center_of_mass[:, None, None, :]
        else:
            centered_atoms14 = atoms14 - center_of_mass

        return center_of_mass, centered_atoms14





    def make_fixed_mask(self,input_str):
        # 初始化mask列表
        mask = []

        # 分割输入字符串为各个区间
        intervals = re.split(r',\s*', input_str)

        # 处理每个区间
        for interval in intervals:
            if re.match(r'[A-Za-z]', interval):
                # 处理保留区间
                match = re.search(r'(\d+)-(\d+)', interval)
                start, end = map(int, match.groups())
                length = end - start + 1
                mask.extend([1] * length)
            else:
                # 处理随机区间
                start, end = map(int, interval.split('-'))
                length = random.randint(start, end)
                mask.extend([0] * length)

        # 转换为Tensor
        mask_tensor = torch.tensor(mask, dtype=torch.int)

        return mask_tensor


    def create_mask(self,L, indices):
        # 初始化一个长度为 L 的列表，所有元素为 0
        mask = [0] * L

        # 根据提供的索引列表，将相应的元素设置为 1
        for index in indices:
            if 0 <= index < L:  # 确保索引在列表长度范围内
                mask[index] = 1

        return mask
    # 解析并处理输入字符串
    def process_input(self,input_str, pdb_dict):
        intervals = re.split(r',\s*', input_str)
        all_positions = []
        # 初始化mask列表
        mask = []
        aatype=[]
        indices_mask=np.zeros_like(pdb_dict['modeled_idx'])

        for interval in intervals:
            if re.match(r'[A-Za-z]', interval):
                # 处理特定链上的残基区间

                chain_char, start, end = re.findall(r'([A-Za-z])(\d+)-(\d+)', interval)[0]
                start, end = int(start), int(end)
                chain_int = du.chain_str_to_int(chain_char)

                # 从pdb_dict中提取相应的原子坐标
                indices = np.where((pdb_dict['chain_index'] == chain_int) &
                                   (pdb_dict['residue_index'] >= start) &
                                   (pdb_dict['residue_index'] <= end))[0]
                # 使用np.take而不是直接索引，以避免TypeError
                positions = np.take(pdb_dict['atom_positions'], indices, axis=0)
                all_positions.append(positions)
                indices_mask[indices]=1
                seqs_motif=np.take(pdb_dict['aatype'], indices, axis=0)
                aatype.append(seqs_motif)

                length = end - start + 1
                mask.extend([1] * length)
            else:
                # 处理随机采样区间

                length = int(interval)
                # 生成随机坐标，假设每个残基有4个原子，每个原子有3个坐标维度
                random_positions = np.random.rand(length, 37, 3)
                all_positions.append(random_positions)
                aatype.append(np.zeros(length))

                mask.extend([0] * length)
        # 转换为Tensor
        mask_tensor = torch.tensor(mask, dtype=torch.int)

        #indices_mask=create_mask(L=pdb_dict['residue_index'].shape[0],indices=indices_mask)
        return np.concatenate(all_positions,axis=0),mask_tensor,np.concatenate(aatype,axis=0),indices_mask

    def test_process_input(self):
        str='E400-510/31,A24-42,A24-42,A64-82,4'
        ref_pdb=''
        ref_data = du.read_pkl(ref_pdb)

@hydra.main(version_base=None, config_path="../configs", config_name="inference_binder")
def main(cfg):
    # make_fixed_mask()
    # 使用cfg对象


    proflow = Proflow(cfg)
    a = 30  # 起始值
    b = 100  # 结束值，不包括在内
    n = 5  # 间隔
    sequence = list(range(a, b, n))
    sample_length=sequence
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(cfg.inference.output_dir, 'sampling_runs')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    param_file = os.path.join(folder_path, f'sampling_params_{timestamp}.txt')
    with open(param_file, 'w') as f:
        f.write(f"Sampling Parameters:\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Sampling Method: {cfg.inference.sampling_method}\n")
        f.write(f"Target: {cfg.inference.target_com_idx}\n")
        f.write(f"Length: random({cfg.inference.min_length}-{cfg.inference.max_length})\n")
        if hasattr(cfg.inference, 'hotspot') and cfg.inference.hotspot:
            f.write(f"Hotspot Residues: {cfg.inference.hotspot}\n")
        if hasattr(cfg.inference, 'fixed_by_chain') and cfg.inference.fixed_by_chain:
            f.write(f"Fixed by Chain: {cfg.inference.fixed_by_chain}\n")
        if hasattr(cfg.inference, 'fixed_com_idx') and cfg.inference.fixed_com_idx:
            f.write(f"Fixed Com Idx: {cfg.inference.fixed_com_idx}\n")
        f.write(f"Number of Samples: {cfg.inference.num_samples}\n")

    for i in range(cfg.inference.num_samples):
        length = random.randint(cfg.inference.min_length, cfg.inference.max_length)
        
        # 根据配置选择使用哪个采样方法
        sampling_method = cfg.inference.sampling_method
      # 会生成 outputs/exp1/config.yaml 和 config.json

        if sampling_method == "hotspot":
            # 使用带热点的采样方法
            outputpath=proflow.sample_binder_bylength_hotspot(
                Target=cfg.inference.target_com_idx, 
                design_num=i, 
                Length=length,
                hotspot=cfg.inference.hotspot if hasattr(cfg.inference, 'hotspot') else [],
                fixed_by_chain=cfg.inference.fixed_by_chain if hasattr(cfg.inference, 'fixed_by_chain') else None,
                design_class_name=cfg.inference.design_class_name,
                base_path=cfg.inference.base_path,
                ref_path=cfg.inference.ref_path
            )
            save_cfg(cfg, outputpath+"/exp1")
        elif sampling_method == "addhotspot":
            # 使用带热点的采样方法
            proflow.sample_DYNbinder_bylength_hotspot(
                Target=cfg.inference.target_com_idx,
                design_num=i,
                Length=length,
                hotspot=cfg.inference.hotspot if hasattr(cfg.inference, 'hotspot') else [],
                fixed_by_chain=cfg.inference.fixed_by_chain if hasattr(cfg.inference, 'fixed_by_chain') else None,
                design_class_name=cfg.inference.design_class_name,
                base_path=cfg.inference.base_path,
                ref_path=cfg.inference.ref_path,
                add_path=cfg.inference.add_path,
                add_target=cfg.inference.add_target_com_idx,
                add_fixed_by_chain=cfg.inference.add_fixed_by_chain
            )

        elif sampling_method == "reference":
            # 使用参考链的采样方法
            proflow.sample_binder_bylength_reference(
                Target=cfg.inference.target_com_idx, 
                design_num=i, 
                Length=length,
                fixed_com_idx=cfg.inference.fixed_com_idx if hasattr(cfg.inference, 'fixed_com_idx') else [],
                hotspot=cfg.inference.hotspot if hasattr(cfg.inference, 'hotspot') else None,
                design_class_name=cfg.inference.design_class_name,
                design_path=cfg.inference.base_path,
                ref_path=cfg.inference.ref_path
            )
        elif sampling_method == "basic":
            # 使用基本的采样方法
            proflow.sample_binder_bylength(
                Target=cfg.inference.target_com_idx, 
                design_num=i, 
                Length=length
            )
        else:
            print(f"Unknown sampling method: {sampling_method}. Using default method.")
            proflow.sample_binder_bylength(
                Target=cfg.inference.target_com_idx, 
                design_num=i, 
                Length=length
            )


if __name__ == '__main__':
#     import re

    set_global_seed(42)

    main()
