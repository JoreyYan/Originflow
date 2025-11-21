from typing import Any
import torch
import torch.nn.functional as F
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel,FlowModel_binder,FlowModel_binder_sidechain,FlowModel_seqdesign
from models import utils as mu
from data.interpolant import Interpolant_10
from data import utils as du
from data.motif_sample import MotifSampler
from models.noise_schedule import OTNoiseSchedule
from data import all_atom
from data import so3_utils



def generate_random_list(total_length, min_size=40):
    assert total_length >= 2 * min_size, "总长度必须至少容纳2段"
    max_parts = min(6, total_length // min_size)
    num_parts = random.randint(2, max_parts)

    base = [min_size] * num_parts
    slack = total_length - num_parts * min_size
    if slack == 0:
        return base  # 已经刚好平均到下限

    # 在 [0, slack] 上取 num_parts-1 个切点，按差分分配富余量
    cuts = sorted(random.randint(0, slack) for _ in range(num_parts - 1))
    chunks = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [slack - cuts[-1]]

    return [b + c for b, c in zip(base, chunks)]


class FlowModule(LightningModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)

        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        self._eps=1e-6



        self.design_task = None


        if 'corrupt_mode' not in self._exp_cfg:
            print('\n  old version no corrupt mode')
            self.model = FlowModel(cfg.model,mode='base')
            # self._exp_cfg.corrupt_mode='base'
        # Set-up vector field prediction model
        else:
            if self._exp_cfg.corrupt_mode == 'binder':
                self.model = FlowModel_binder(cfg.model)
            elif self._exp_cfg.corrupt_mode == 'motif':
                self.model = FlowModel(cfg.model)

            elif self._exp_cfg.corrupt_mode == 'sidechain':
                # self.pretrained_part  = FlowModel_binder(cfg.model)
                self.model = FlowModel_binder_sidechain(cfg.model)
            elif self._exp_cfg.corrupt_mode == 'base':   # base for mononer or complex
                self.model = FlowModel(cfg.model,mode='base')

            elif self._exp_cfg.corrupt_mode == 'base_ss':   # base for mononer or complex
                self.model = FlowModel(cfg.model,mode='base_ss')


            elif self._exp_cfg.corrupt_mode == 'fbb':   # base for mononer or complex
                self.model = FlowModel_seqdesign(cfg.model)




        # Set-up interpolant
        self.interpolant = Interpolant_10(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        # Noise schedule
        self.noise_schedule=OTNoiseSchedule()


    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip))

        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * 3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * 3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
                t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        if torch.isnan(se3_vf_loss).any():
            raise ValueError('NaN loss encountered')
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape

        samples = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
        )[0][-1].numpy()

        batch_metrics = []
        for i in range(num_batch):

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order['CA']
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)

    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
    ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k, v in batch_losses.items()
        }
        for k, v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(
                t, loss_dict, loss_name=loss_name)
            for k, v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = (
                total_losses[self._exp_cfg.training.loss]
                + total_losses['auxiliary_loss']
        )
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
    def predict_step_FUNC(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        num_batch=2

        self.interpolant._sample_cfg.num_timesteps = 500

        self._sample_write_dir = '/home/junyu/project/monomer_test/homo_heto/monomder_ss/' + str(
            self.interpolant._sample_cfg.num_timesteps) + '/sample/'
        os.makedirs(self._sample_write_dir, exist_ok=True)

        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()

        sample_id = 0
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        sample_length = batch['num_res'].item()



        sample_id = batch['sample_id'].item()
        self.interpolant.set_device(device)
        res_mask=torch.ones((sample_id,sample_length),device=device)
        if self._exp_cfg.corrupt_mode == 'motif':
            samples = self.val_complex( num_batch, sample_length,self._exp_cfg.corrupt_mode)
            # samples, fixed_mask = self.val_motif(batch, sample_id, sample_length, res_mask)
            X = samples[0]
            C = samples[1]
            S = samples[2]
        elif self._exp_cfg.corrupt_mode == 'binder':
            samples = self.val_complex(num_batch, sample_length,self._exp_cfg.corrupt_mode)
            #samples, noisy_batch = self.val_binder(batch, num_batch, num_res, res_mask)
            X = samples[0]
            C = samples[1]
            S = samples[2]
            bf = samples[3]




        for i in range(num_batch):
            if self._exp_cfg.corrupt_mode == 'binder':

                p = Protein.from_XCSB(X, C, S, bf)
                p.to_PDB(os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))


            else:
                p = Protein.from_XCS(X[i].unsqueeze(0), C[i].unsqueeze(0), S[i].unsqueeze(0))
                p.to_PDB(os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))

    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        num_batch = 1
        sample_length = batch['num_res'].item()

        sample_id = batch['sample_id'].item()
        self.interpolant.set_device(device)

        if self.design_task == 'sym':
            self.predict_step_sym(batch, batch_idx)
        elif self.design_task=='monomer':
            if self._exp_cfg.corrupt_mode == 'base':
                self.predict_step_base(batch, batch_idx)
            elif  self._exp_cfg.corrupt_mode == 'base_ss':
                print('corrupt mode is base_ss')
                self.predict_step_base(batch, batch_idx)
            elif  self._exp_cfg.corrupt_mode == 'motif':
                print('corrupt mode is motif')
                self.predict_step_base(batch, batch_idx)

            else:
                print('corrupt mode not defined')

        elif self.design_task=='homomer':

            self.predict_step_homo(batch, batch_idx)

        elif self.design_task=='motif':

            self.predict_step_motif(batch, batch_idx)
        elif self.design_task=='monomer_ss':

            self.predict_step_base_ss(batch, batch_idx)
        elif self.design_task=='base_ss':
            self.predict_step_base_ss(batch, batch_idx)
        else:
            self.predict_step_FUNC(batch, batch_idx)

            samples = self.val_complex(num_batch, sample_length, self._exp_cfg.corrupt_mode)

    def predict_step_base(self, batch, batch_idx):

        print('design for  , ',self.design_task)

        methods=self._infer_cfg.interpolant.sampling.methods
        num_timesteps=self._infer_cfg.interpolant.sampling.num_timesteps

        self.interpolant._sample_cfg.num_timesteps=num_timesteps
        print(f'design case in, {num_timesteps} steps', )

        self._sample_write_dir=(self._output_dir+f'/{self.design_task}_{methods}_temp'+str(self.interpolant._cfg.temp)+'_'
                                +str(num_timesteps)+'_0819_clean/')


        os.makedirs(self._sample_write_dir, exist_ok=True)
        print('write in ,', self._sample_write_dir)


        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()

        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        num_batch = 1



        samples= interpolant.heun_sample(
            num_batch, sample_length, self.model
        )
        X = samples[0]
        C = samples[1]
        S = samples[2]

        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            C_i = C[i].unsqueeze(0)
            S_i = S[i].unsqueeze(0)

            p = Protein.from_XCS(X_i, C_i, S_i)
            p.to_PDB(os.path.join(
                self._sample_write_dir,
                f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))


    def predict_step_base_ss(self, batch, batch_idx):
        '''
        design with ss
        '''

        print('design for base , ',self.design_task)
        self.interpolant._sample_cfg.num_timesteps=500
        methods='cvode_ss'

        pdb_name=self._cfg.pdb_name
        ss_list_str=self._cfg.ss_list_str

        print('design: ', pdb_name,ss_list_str)

        self._sample_write_dir=self._output_dir+f'/32_{methods}_{pdb_name}_temp'+str(self.interpolant._cfg.temp)+'_'+str(self.interpolant._sample_cfg.num_timesteps)+'/'
        os.makedirs(self._sample_write_dir, exist_ok=True)
        print('write in ,', self._sample_write_dir)


        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()

        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        num_batch = 1
        ss= eval(ss_list_str)


        # 随机生成长度在60到200之间


        if         methods=='cvode':
            samples= interpolant.sample(
                num_batch, sample_length, self.model
            )
        else:

            samples = self.interpolant.hybrid_Complex_sample(
                num_batch,
                [len(ss)],
                self.model,
                ss,
            )

        X = samples[0]
        C = samples[1]
        S = samples[2]

        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            C_i = C[i].unsqueeze(0)
            S_i = S[i].unsqueeze(0)

            p = Protein.from_XCS(X_i, C_i, S_i)
            p.to_PDB(os.path.join(
                self._sample_write_dir,
                f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))

    def predict_step_homo(self, batch, batch_idx):

        print('design for  , ', self.design_task)

        methods = self._infer_cfg.interpolant.sampling.methods
        num_timesteps = self._infer_cfg.interpolant.sampling.num_timesteps

        self.interpolant._sample_cfg.num_timesteps = num_timesteps
        print(f'design case in, {num_timesteps} steps', )

        self._sample_write_dir = (self._output_dir + f'/32_{methods}_temp' + str(self.interpolant._cfg.temp) + '_'
                                  + str(num_timesteps) + '/')

        os.makedirs(self._sample_write_dir, exist_ok=True)
        print('write in ,', self._sample_write_dir)






        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()

        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        num_batch = 1

        numres=generate_random_list(sample_length)

        samples = self.interpolant.hybrid_Complex_sample(
            num_batch,
            numres,
            self.model,
        )

        X = samples[0]
        C = samples[1]
        S = samples[2]

        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            C_i = C[i].unsqueeze(0)
            S_i = S[i].unsqueeze(0)

            p = Protein.from_XCS(X_i, C_i, S_i)
            p.to_PDB(os.path.join(
                self._sample_write_dir,
                f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))
    def predict_step_sym(self, batch, batch_idx):
        print(batch_idx, 'sym')


        self.interpolant._sample_cfg.num_timesteps=500
        methods='cvode_sym'

        sym_mode=self._cfg.inference.sym
        print('predict_step_sym: ', sym_mode)
        self._sample_write_dir=self._output_dir+f'/sym_{sym_mode}_{methods}_try_temp'+str(self.interpolant._cfg.temp)+'_'+str(self.interpolant._sample_cfg.num_timesteps)+'/'
        os.makedirs(self._sample_write_dir, exist_ok=True)



        from chroma.data.protein import Protein
        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        num_batch = 1



        samples = self.interpolant.hybrid_Complex_sym_sample(
            num_batch,
            [sample_length],
            self.model,
            symmetry=sym_mode,
        )

        X = samples[0]
        C = samples[1]
        S = samples[2]

        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            C_i = C[i].unsqueeze(0)
            S_i = S[i].unsqueeze(0)

            p = Protein.from_XCS(X_i, C_i, S_i)
            p.to_PDB(os.path.join(
                self._sample_write_dir,
                f'sample_{i}_idx_{batch_idx}_len_{sample_length}.pdb'))
    def predict_step_motif(self, batch, batch_idx):

        print('design for  , ',self.design_task)

        methods=self._infer_cfg.interpolant.sampling.methods
        num_timesteps=self._infer_cfg.interpolant.sampling.num_timesteps

        self.interpolant._sample_cfg.num_timesteps=num_timesteps
        print(f'design case in, {num_timesteps} steps', )


        print(self._output_dir)


        ref_pdb = '/home/junyu/project/motif/rf_pdb_pkl/5tpn.pkl'
        domain=ref_pdb.split('/')[-1].split('.')[0]
        ref_data = du.read_pkl(ref_pdb)
        input_str="10-40,A163-181,10-40"

        # 使用当前时间戳设置种子值
        # random.seed(time.time())
        total_length=random.randint(50, 75)



        sampler = MotifSampler(input_str, total_length)
        results = sampler.get_results()
        print(f"Letter segments: {results['letter_segments']}")
        print(f"Number segments: {results['number_segments']}")
        print(f"Total motif length: {results['total_motif_length']}")
        print(f"Random sample total length: {results['random_sample_total_length']}")
        print(f"Sampled lengths: {results['sampled_lengths']}")
        final_output = sampler.get_final_output()
        print(f"Final output: {final_output}")




        designname=ref_pdb.split('/')[-1].split('.')[0]
        self._sample_write_dir=self._output_dir+f'/trysampleupdate_{designname}_{methods}_temp'+str(self.interpolant._cfg.temp)+'_'+str(self.interpolant._sample_cfg.num_timesteps)+'/'

        #self._sample_write_dir='/home/junyu/project/monomer_test/base_neigh/rcsb_ipagnnneigh_cluster_motif_updateall_reeidx_homo_heto/2024-03-12_12-52-05/last_256/rcsb/motif_1bcf/'
        os.makedirs(self._sample_write_dir, exist_ok=True)
        os.makedirs(self._sample_write_dir+'/native/', exist_ok=True)
        os.makedirs(self._sample_write_dir+'/motif_masks/', exist_ok=True)


        from .Proflow import process_input
        from chroma.data.protein import Protein
        import tqdm

        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        # native structure
        positions = np.take(ref_data['atom_positions'], ref_data['modeled_idx'], axis=0)[..., [0, 1, 2, 4], :]
        chain_index = np.take(ref_data['chain_index'], ref_data['modeled_idx'], axis=0)
        aatype = np.take(ref_data['aatype'], ref_data['modeled_idx'], axis=0)
        p = Protein.from_XCS(torch.tensor(positions).unsqueeze(0), torch.tensor(chain_index).unsqueeze(0),
                             torch.tensor(aatype).unsqueeze(0))
        p.to_PDB(self._sample_write_dir + f'/native/{domain}_motif_native.pdb')

        _, _, _, indices_mask = process_input(final_output, ref_data)
        np.savetxt(self._sample_write_dir + f'/motif_masks/motif_native.npy', indices_mask)

        for i in tqdm.tqdm(range(10)):
            total_length = random.randint(50, 75)
            sampler = MotifSampler(input_str, total_length)
            results = sampler.get_results()
            print(f"Letter segments: {results['letter_segments']}")
            print(f"Number segments: {results['number_segments']}")
            print(f"Total motif length: {results['total_motif_length']}")
            print(f"Random sample total length: {results['random_sample_total_length']}")
            print(f"Sampled lengths: {results['sampled_lengths']}")
            final_output = sampler.get_final_output()
            print(f"Final output: {final_output}")

            init_motif, fixed_mask, aa_motifed, indices_mask = process_input(final_output, ref_data)



            chain_idx = torch.ones_like(fixed_mask).unsqueeze(0)
            fixed_mask = fixed_mask.unsqueeze(0)
            bbatoms = torch.tensor(init_motif)[..., [0, 1, 2, 4], :].unsqueeze(0).to(self.device).float()
            sample_length = bbatoms.shape[1]



            X, C, S = self.interpolant.try_motif_sample(
                1,
                [sample_length],
                self.model,
                chain_idx=chain_idx.to(self.device),
                native_X=bbatoms.to(self.device),
                mode='motif',
                fixed_mask=fixed_mask.to(self.device),

            )

            native_aatype=torch.tensor(aa_motifed).unsqueeze(0).to(S.device)
            S=native_aatype*fixed_mask.to(S.device)
            p = Protein.from_XCS(X, C, S, )


            # 输出到文本文件
            p.to_PDB(self._sample_write_dir+f'/{domain}_motif' + str(i) + '.pdb')
            np.savetxt(self._sample_write_dir + f'/motif_masks/{domain}_motif' + str(i) +'_mask.npy', fixed_mask.cpu().numpy())

            # positions = torch.nonzero(fixed_mask.squeeze(0) == 1).squeeze(-1)
            # # 转换为列表
            # positions_list = positions.tolist()
            # # 转换为逗号分隔的字符串
            # positions_str = ', '.join(map(str, positions_list))
            # with open(self._sample_write_dir + '/5yui_motif_info.txt', 'w') as f:
            #
            #         f.write(f"> 5yui_motif, fixed area \n")
            #         f.write(f"> in native \n")
            #         f.write(f"> {parama}  \n")
            #         f.write(f"> in design \n")
            #         f.write(f"{positions_str}\n")


    def predict_step_motif_fixlength(self, batch, batch_idx):

        print('design for  , ',self.design_task)

        methods=self._infer_cfg.interpolant.sampling.methods
        num_timesteps=self._infer_cfg.interpolant.sampling.num_timesteps

        self.interpolant._sample_cfg.num_timesteps=num_timesteps
        print(f'design case in, {num_timesteps} steps', )


        print(self._output_dir)


        ref_pdb = '/home/junyu/project/motif/rf_pdb_pkl/5tpn.pkl'
        domain=ref_pdb.split('/')[-1].split('.')[0]
        ref_data = du.read_pkl(ref_pdb)
        input_str="10-40,A163-181,10-40"

        # 使用当前时间戳设置种子值
        # random.seed(time.time())
        total_length=random.randint(50, 75)



        sampler = MotifSampler(input_str, total_length)
        results = sampler.get_results()
        print(f"Letter segments: {results['letter_segments']}")
        print(f"Number segments: {results['number_segments']}")
        print(f"Total motif length: {results['total_motif_length']}")
        print(f"Random sample total length: {results['random_sample_total_length']}")
        print(f"Sampled lengths: {results['sampled_lengths']}")
        final_output = sampler.get_final_output()
        print(f"Final output: {final_output}")




        designname=ref_pdb.split('/')[-1].split('.')[0]
        self._sample_write_dir=self._output_dir+f'/motif_motifdesign_{designname}_{methods}_temp'+str(self.interpolant._cfg.temp)+'_'+str(self.interpolant._sample_cfg.num_timesteps)+'/'

        #self._sample_write_dir='/home/junyu/project/monomer_test/base_neigh/rcsb_ipagnnneigh_cluster_motif_updateall_reeidx_homo_heto/2024-03-12_12-52-05/last_256/rcsb/motif_1bcf/'
        os.makedirs(self._sample_write_dir, exist_ok=True)
        os.makedirs(self._sample_write_dir+'/native/', exist_ok=True)
        os.makedirs(self._sample_write_dir+'/motif_masks/', exist_ok=True)


        from .Proflow import process_input
        from chroma.data.protein import Protein
        import tqdm

        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        # native structure
        positions = np.take(ref_data['atom_positions'], ref_data['modeled_idx'], axis=0)[..., [0, 1, 2, 4], :]
        chain_index = np.take(ref_data['chain_index'], ref_data['modeled_idx'], axis=0)
        aatype = np.take(ref_data['aatype'], ref_data['modeled_idx'], axis=0)
        p = Protein.from_XCS(torch.tensor(positions).unsqueeze(0), torch.tensor(chain_index).unsqueeze(0),
                             torch.tensor(aatype).unsqueeze(0))
        p.to_PDB(self._sample_write_dir + f'/native/{domain}_motif_native.pdb')

        _, _, _, indices_mask = process_input(final_output, ref_data)
        np.savetxt(self._sample_write_dir + f'/motif_masks/motif_native.npy', indices_mask)

        for i in tqdm.tqdm(range(10)):
            total_length = random.randint(50, 75)
            sampler = MotifSampler(input_str, total_length)
            results = sampler.get_results()
            print(f"Letter segments: {results['letter_segments']}")
            print(f"Number segments: {results['number_segments']}")
            print(f"Total motif length: {results['total_motif_length']}")
            print(f"Random sample total length: {results['random_sample_total_length']}")
            print(f"Sampled lengths: {results['sampled_lengths']}")
            final_output = sampler.get_final_output()
            print(f"Final output: {final_output}")

            init_motif, fixed_mask, aa_motifed, indices_mask = process_input(final_output, ref_data)



            chain_idx = torch.ones_like(fixed_mask).unsqueeze(0)
            fixed_mask = fixed_mask.unsqueeze(0)
            bbatoms = torch.tensor(init_motif)[..., [0, 1, 2, 4], :].unsqueeze(0).to(self.device).float()
            sample_length = bbatoms.shape[1]



            X, C, S = self.interpolant.hybrid_motif_sample(
                1,
                [sample_length],
                self.model,
                chain_idx=chain_idx.to(self.device),
                native_X=bbatoms.to(self.device),
                mode='motif',
                fixed_mask=fixed_mask.to(self.device),

            )

            native_aatype=torch.tensor(aa_motifed).unsqueeze(0).to(S.device)
            S=native_aatype*fixed_mask.to(S.device)
            p = Protein.from_XCS(X, C, S, )


            # 输出到文本文件
            p.to_PDB(self._sample_write_dir+f'/{domain}_motif' + str(i) + '.pdb')
            np.savetxt(self._sample_write_dir + f'/motif_masks/{domain}_motif' + str(i) +'.npy', fixed_mask.cpu().numpy())

            # positions = torch.nonzero(fixed_mask.squeeze(0) == 1).squeeze(-1)
            # # 转换为列表
            # positions_list = positions.tolist()
            # # 转换为逗号分隔的字符串
            # positions_str = ', '.join(map(str, positions_list))
            # with open(self._sample_write_dir + '/5yui_motif_info.txt', 'w') as f:
            #
            #         f.write(f"> 5yui_motif, fixed area \n")
            #         f.write(f"> in native \n")
            #         f.write(f"> {parama}  \n")
            #         f.write(f"> in design \n")
            #         f.write(f"{positions_str}\n")
