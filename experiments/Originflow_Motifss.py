import os
import time
import random
import numpy as np
import hydra
import torch
import GPUtil
import pandas as pd
import tqdm
from datetime import datetime
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from models.flow_module import FlowModule
from data import utils as du
from data.m2 import MotifSampler, MotifSamplerMultiChain
from data.pdb_dataloader import mapping_dict, vectorized_mapping
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


class Sampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        try:
            ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

            del ckpt_cfg.model
            # Set-up config.
            OmegaConf.set_struct(cfg, False)
            OmegaConf.set_struct(ckpt_cfg, False)
            cfg = OmegaConf.merge(cfg, ckpt_cfg)
        except:
            OmegaConf.set_struct(cfg, False)

        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        # Simplified output path: directly to output/motif_ss
        self._output_dir = self._infer_cfg.output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Base output directory: {self._output_dir}')

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path, strict=True
        )
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir
        self._flow_module.interpolant._cfg.temp = self._infer_cfg.lam
        self._flow_module.interpolant._sample_cfg = self._flow_module._infer_cfg.interpolant.sampling

    def parse_ref_ss_file(self, ss_file_path):
        """Parse ref_ss.txt file to extract sequence and SS.

        Returns:
            tuple: (sequence_str, ss_str)
        """
        with open(ss_file_path, 'r') as f:
            lines = f.readlines()

        # Format: >name_sequence\nSEQUENCE\n>name_secondary_structure\nSS
        sequence = lines[1].strip()
        ss = lines[3].strip()
        return sequence, ss

    def extract_motif_region(self, motif_def, sequence, ss):
        """Extract motif sequence and SS based on definition like 'A41-56'.

        Args:
            motif_def: str, like "A41-56" or "41-56"
            sequence: full sequence string
            ss: full SS string

        Returns:
            tuple: (motif_sequence, motif_ss)
        """
        # Parse motif definition
        import re
        match = re.match(r'([A-Z])?(\d+)-(\d+)', motif_def)
        if match:
            chain_id, start, end = match.groups()
            start_idx = int(start) - 1  # Convert to 0-based index
            end_idx = int(end)

            motif_seq = sequence[start_idx:end_idx]
            motif_ss = ss[start_idx:end_idx]
            return motif_seq, motif_ss
        else:
            raise ValueError(f"Invalid motif definition: {motif_def}")

    def run_sampling(self, pdb_input_list, pkl_folder_path):
        devices = GPUtil.getAvailable(
            order='memory', limit=8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
            precision=32,
        )

        # trainer.predict(self._flow_module, dataloaders=dataloader)

        # Generate timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log.info(f"Run timestamp: {timestamp}")

        # Get sampling method from config
        sampling_method = getattr(self._infer_cfg.interpolant.sampling, 'sampling_method', 'euler')
        log.info(f"Using sampling method: {sampling_method}")
        log.info(f"Total number of designs to process: {len(pdb_input_list)}")

        for idx, (name, input_combined) in enumerate(pdb_input_list):
            # Parse input: maintain original order of SS and motif regions
            # e.g., "HHHHHHHHHHHH,A20-31,HHHHHHHHHHHH"
            # Result: scaffold1 + motif_ss + scaffold2
            # SS part: contains only letters (H, E, C)
            # Motif part: contains numbers (like A41-56, 10-25)

            parts = input_combined.split(',')

            # Read ref_ss.txt to get motif SS
            ref_ss_path = os.path.join(pkl_folder_path, name, 'preprocessed', f"{name}_ref_ss.txt")
            ref_sequence, ref_ss = self.parse_ref_ss_file(ref_ss_path)

            log.info(f"Processing [{idx}] {name}:")
            log.info(f"  Input structure: {input_combined}")

            # Process each part in order, building SS and input_str simultaneously
            full_ss_parts = []
            modified_input_parts = []

            for part in parts:
                part = part.strip()
                # Check if part contains any digit (motif region)
                if any(char.isdigit() for char in part):
                    # This is a motif region, extract its SS from ref_ss.txt
                    motif_seq, motif_ss = self.extract_motif_region(part, ref_sequence, ref_ss)
                    full_ss_parts.append(motif_ss)
                    modified_input_parts.append(part)
                    log.info(f"  Motif {part}: SS={motif_ss} (length={len(motif_ss)})")
                else:
                    # This is a scaffold SS string
                    full_ss_parts.append(part)
                    modified_input_parts.append(str(len(part)))
                    log.info(f"  Scaffold SS: {part} (length={len(part)})")

            # Construct full SS by concatenating in order
            full_ss_str = ''.join(full_ss_parts)
            ss_list = list(full_ss_str)
            ss_mapped = vectorized_mapping(np.array(ss_list))

            # Modify input_str: replace SS strings with their lengths
            # Original: "HHHHHHHHHHHH,A20-31,HHHHHHHHHHHH"
            # Modified: "12,A20-31,12"
            modified_input_str = ','.join(modified_input_parts)

            log.info(f"  Modified input_str: {modified_input_str}")
            log.info(f"  Full SS: {full_ss_str}")
            log.info(f"  Full SS length: {len(full_ss_str)}")

            # New path structure: {pkl_folder_path}/{name}/preprocessed/{name}.pkl
            ref_pdb = os.path.join(pkl_folder_path, name, 'preprocessed', f"{name}.pkl")
            ref_data = du.read_pkl(ref_pdb)
            domain = name

            # Use modified input_str
            sampler = MotifSamplerMultiChain(modified_input_str)
            results = sampler.get_results()
            final_output = sampler.get_final_output()

            # Create output directory for this PDB with timestamp and row index
            self._flow_module._output_dir = os.path.join(self._output_dir, f'{domain}__motif_ss_{timestamp}_{idx}')
            os.makedirs(self._flow_module._output_dir, exist_ok=True)
            os.makedirs(self._flow_module._output_dir + '/native/', exist_ok=True)
            os.makedirs(self._flow_module._output_dir + '/motif_masks/', exist_ok=True)
            log.info(f"  Output directory: {self._flow_module._output_dir}")

            # Save config to this PDB's output directory
            config_path = os.path.join(self._flow_module._output_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'  Saved config to {config_path}')

            from help.test_process_mc_input import process_input
            from chroma.data.protein import Protein

            device = f'cuda:{torch.cuda.current_device()}'
            self._flow_module.interpolant.set_device(device)

            positions = np.take(ref_data['atom_positions'], ref_data['modeled_idx'], axis=0)[..., [0, 1, 2, 4], :]
            chain_index = np.take(ref_data['chain_index'], ref_data['modeled_idx'], axis=0)
            aatype = np.take(ref_data['aatype'], ref_data['modeled_idx'], axis=0)

            # Get motif mask from process_input
            _, _, _, indices_mask, _ = process_input(final_output, ref_data)

            # Save motif mask
            np.savetxt(os.path.join(self._flow_module._output_dir + '/motif_masks/', f'{domain}_motif_native_mask.npy'),
                       indices_mask)

            # Save native structure (full structure, not just motif)
            p = Protein.from_XCS(
                torch.tensor(positions).unsqueeze(0),
                torch.tensor(chain_index).unsqueeze(0),
                torch.tensor(aatype).unsqueeze(0)
            )
            native_motif_path = os.path.join(self._flow_module._output_dir + '/native/', f'{domain}_native_motif.pdb')
            p.to_PDB(native_motif_path)
            log.info(f"  Saved native structure to native/")

            # Get number of designs from config
            num_designs = self._samples_cfg.samples_per_length
            log.info(f"Generating {num_designs} designs for {domain}")

            for i in tqdm.tqdm(range(num_designs)):
                # Resample with the modified input
                sampler = MotifSamplerMultiChain(modified_input_str)
                results = sampler.get_results()
                final_output = sampler.get_final_output()

                init_motif, fixed_mask, aa_motifed, indices_mask, chain_ids = process_input(final_output, ref_data)

                chain_idx = torch.tensor(chain_ids).unsqueeze(0)

                fixed_mask = fixed_mask.unsqueeze(0)
                bbatoms = torch.tensor(init_motif)[..., [0, 1, 2, 4], :].unsqueeze(0).to(device).float()
                sample_length =bbatoms.shape[1]

                # Prepare SS tensor
                ss_tensor = torch.tensor(ss_mapped).unsqueeze(0).to(device)

                X, C, S = self._flow_module.interpolant.hybrid_motif_sample(
                    1,
                    [sample_length],
                    self._flow_module.model,
                    chain_idx=chain_idx.to(device),
                    native_X=bbatoms.to(device),
                    rgadd=1.1,
                    ss=ss_tensor,
                    fixed_mask=fixed_mask.to(device),
                    sampling_method=sampling_method,
                )

                native_aatype = torch.tensor(aa_motifed).unsqueeze(0).to(S.device)
                S = native_aatype * fixed_mask.to(S.device)
                p = Protein.from_XCS(X, C, S)

                p.to_PDB(os.path.join(self._flow_module._output_dir, f'{domain}_motif_ss_{i}.pdb'))
                np.savetxt(os.path.join(self._flow_module._output_dir + '/motif_masks/', f'{domain}_motif_ss_{i}_mask.npy'),
                           fixed_mask.cpu().numpy())
                np.savetxt(os.path.join(self._flow_module._output_dir + '/motif_masks/', f'{domain}_motif_ss_{i}_Chain_mask.npy'),
                           C.squeeze(0).cpu().numpy())


@hydra.main(version_base=None, config_path="../configs", config_name="inference_motif_ss")
def run(cfg: DictConfig) -> None:
    lam = [1]
    for l0 in lam:
        cfg.inference.lam = l0

        # Read model checkpoint.
        log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
        start_time = time.time()

        sampler = Sampler(cfg)

        # Read ref_pdb and input_str from CSV
        # Note: Input column format: "scaffold_ss,motif_region"
        # e.g., "CEEEEEECCCCCCEEEEEECCCHHHHHHHHHHHHHCCCCCC,A41-56"
        csv_path = "../examples/1fna_motifSS_PDB_Input_Data.csv"
        # New path structure: rf_ss_pdb/{PDB_NAME}/preprocessed/{PDB_NAME}.pkl
        pkl_folder_path = "../examples/rf_ss_pdb/"
        df = pd.read_csv(csv_path)
        pdb_input_list = df[['Name', 'Input']].values.tolist()

        sampler.run_sampling(pdb_input_list, pkl_folder_path)
        elapsed_time = time.time() - start_time
        log.info(f'Finished in {elapsed_time:.2f}s')


if __name__ == '__main__':
    run()
