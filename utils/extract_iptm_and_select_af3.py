import os
import json
import csv
import pandas as pd

def convert_cif_to_pdb(cif_path, pdb_path):
    from Bio.PDB import MMCIFParser, PDBIO
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)

def extract_and_filter_iptm(root_dir):
    # 1. 提取所有json中的分数，写入 summary_confidence_scores.csv
    rows = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # if "seed_101/predictions" in dirpath.replace("\\", "/"):  # 兼容 Windows 路径
            for filename in filenames:
                if filename.endswith(".json"):
                    json_path = os.path.join(dirpath, filename)
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        row = [
                            filename,
                            data.get("plddt", None),
                            data.get("gpde", None),
                            data.get("ptm", None),
                            data.get("iptm", None),
                        ]
                        rows.append(row)
                    except Exception as e:
                        print(f"Error reading {json_path}: {e}")

    summary_csv = os.path.join(root_dir, "summary_confidence_scores.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "plddt", "gpde", "ptm", "iptm"])
        writer.writerows(rows)
    print(f"已保存所有分数到: {summary_csv}")

    # 2. 读取csv，筛选iptm>0.6，按prefix分组取iptm最大，写入filtered_best_by_prefix.csv
    df = pd.read_csv(summary_csv)
    filtered_df = df[df["iptm"] > 0.8].copy()
    filtered_df["prefix"] = filtered_df["filename"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    best_df = filtered_df.sort_values("iptm", ascending=False).drop_duplicates("prefix")
    best_df = best_df.drop(columns=["prefix"])
    filtered_csv = os.path.join(root_dir, "filtered_best_by_prefix.csv")
    best_df.to_csv(filtered_csv, index=False)
    print(f"已保存筛选后的csv到: {filtered_csv}")

    # 3. 将筛选csv中的cif转pdb，输出到 selection_1st 目录
    output_dir = os.path.join(root_dir, "selection_1st/pdb/")
    os.makedirs(output_dir, exist_ok=True)
    prediction_dir = root_dir  # 假设cif文件就在root_dir下的子目录
    for fname in best_df["filename"]:
        # 1. 提取前缀（目录名）: gcpr__1_design_1_
        prefix = "_".join(fname.split("_")[:10])
        # 2. 提取 seed 目录
        seed = [x for x in fname.split("_") if x.startswith("seed")][0]
        # 3. 构建 .cif 文件名
        cif_name = fname.replace("_summary_confidence", "").replace(".json", ".cif")
        # 4. 拼接原始路径
        cif_path = os.path.join(prediction_dir, prefix, seed+'_101', "predictions", cif_name)
        if os.path.exists(cif_path):
            pdb_name = cif_name.replace(".cif", ".pdb")
            pdb_path = os.path.join(output_dir, pdb_name)
            try:
                convert_cif_to_pdb(cif_path, pdb_path)
                print(f"✅ 转换成功：{pdb_name}")
            except Exception as e:
                print(f"⚠️ 转换失败：{cif_name} → 错误：{e}")
        else:
            print(f"❌ 未找到 CIF 文件：{cif_path}")
    print("✅ 所有文件处理完毕。")

if __name__ == "__main__":
    root_dir = "/home/junyu/project/binder_target/gcpr1/preprocessed/__with_hotspot_fixed_chain_1/binder_by_sample_binder_bylength_hotspot_from90/ESMfoldmini_ca_seq8/structures_proteinx/selection_1st/AFPDB/"  # 替换为你包含 gpcr1 文件夹的上级目录

    extract_and_filter_iptm(root_dir)