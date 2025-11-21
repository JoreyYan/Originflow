#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_5.out

source activate SE3nv

folder_with_pdbs="//home/junyu/project/sym/base_neigh/ckpt/ss_sym/last_H800/motif_sym/sym_c6_cvode_sym_tempNone_500/"

output_dir="/home/junyu/project/sym/base_neigh/ckpt/ss_sym/last_H800/motif_sym/sym_c6_cvode_sym_tempNone_500/MPNN_results/"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi


path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$output_dir"/assigned_pdbs.jsonl"
path_for_fixed_positions=$output_dir"/fixed_pdbs.jsonl"
path_for_tied_positions=$output_dir"/tied_pdbs.jsonl"
chains_to_design="A B C D E F"
fixed_positions=
length=90

# 初始化空的tied_positions
tied_positions=""

# 生成 1 到 length 的序列，并将其转为字符串
sequence=$(seq -s " " 1 $length)

# 根据 chains_to_design 的长度生成对应的序列
for chain in $chains_to_design; do
  if [ -z "$tied_positions" ]; then
    tied_positions="$sequence"
  else
    tied_positions="$tied_positions, $sequence"
  fi
done


python helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"

python helper_scripts/make_fixed_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_fixed_positions --chain_list "$chains_to_design" --position_list "$fixed_positions"

python helper_scripts/make_tied_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_tied_positions --chain_list "$chains_to_design" --position_list "$tied_positions"

python protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --fixed_positions_jsonl $path_for_fixed_positions \
        --tied_positions_jsonl $path_for_tied_positions \
        --out_folder $output_dir \
        --num_seq_per_target 8 \
        --sampling_temp "0.1" \
        --seed 0 \
        --batch_size 1
