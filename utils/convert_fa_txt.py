import os
import json

input_fa_path =  "/home/junyu/project/sym/motif_sym/paper_design/c/c3/MPNN_results/seqs/af_strucutre/selection/MPNN_results/seqs/c3_025_07_14_22_00.fa"
fa_filename = os.path.splitext(os.path.basename(input_fa_path))[0]
output_json_dir = os.path.join(
    "/home/junyu/project/sym/motif_sym/paper_design/c/c3/MPNN_results/seqs/af_strucutre/selection/MPNN_results//seqs/af3_json_inputs", fa_filename)
os.makedirs(output_json_dir, exist_ok=True)

# 读取 fasta 文件，跳过第一个参考序列
with open(input_fa_path, "r") as f:
    lines = f.read().splitlines()

sequences = []
current_seq = ""
skipped_first = False
for line in lines:
    if line.startswith(">"):
        if current_seq:
            if not skipped_first:
                skipped_first = True
            else:
                sequences.append(current_seq)
            current_seq = ""
    else:
        current_seq += line.strip()
if current_seq and skipped_first:
    sequences.append(current_seq)

# 处理每一个序列组
for idx, full_seq in enumerate(sequences):
    chain_seqs = full_seq.split("/")  # 按链分割
    unique_chain_seq = chain_seqs[0]  # 任取一个链序列

    # 生成每个链一个 proteinChain，并为每个链添加独特的 ID
    sequence_entries = []
    for i, chain_seq in enumerate(chain_seqs):
        # 生成唯一 ID，从 'A' 开始
        protein_id = chr(ord('A') + i)

        sequence_entries.append({
            "protein": {
                "id": protein_id,  # 给每个链一个唯一 ID
                "sequence": chain_seq,

            }
        })

    json_data = {
        "name": f"{fa_filename}_seq_{idx}",
        "sequences": sequence_entries,  # 包含多个 proteinChain
        "modelSeeds": [0],
        "dialect": "alphafold3",
        "version": 1
    }

    output_path = os.path.join(output_json_dir, f"{fa_filename}_seq_{idx}.json")
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

print(f"✅ 生成 {len(sequences)} 个 JSON，路径：{output_json_dir}")
