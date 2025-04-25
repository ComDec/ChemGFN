import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


class ParallelProcessor:
    def __init__(self, input_npy, output_dir, chunk_size=100000):
        self.input_npy = Path(input_npy)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化分块目录
        self.chunk_dir = self.output_dir / "chunks"
        self.chunk_dir.mkdir(exist_ok=True)

        # 创建处理状态记录
        self.status_db = self.output_dir / "processing.db"
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.status_db)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks
                       (id INT PRIMARY KEY, file TEXT, status TEXT)"""
        )
        conn.commit()
        conn.close()

    def prepare_chunks(self):
        """将npy文件预处理为多个块文件"""
        smiles_array = np.load(self.input_npy, allow_pickle=True)
        total = len(smiles_array)

        conn = sqlite3.connect(self.status_db)
        for i in range(0, total, self.chunk_size):
            chunk_file = self.chunk_dir / f"chunk_{i//self.chunk_size}.npy"
            if not chunk_file.exists():
                np.save(chunk_file, smiles_array[i : i + self.chunk_size])

            conn.execute(
                "INSERT OR IGNORE INTO chunks VALUES (?, ?, ?)",
                (i // self.chunk_size, str(chunk_file), "pending"),
            )
        conn.commit()
        conn.close()

    def process_single_chunk(self, chunk_file, chunk_id):
        """处理单个块的函数（将被parallel调用）"""
        # 加载块数据
        chunk = np.load(chunk_file, allow_pickle=True)

        # 处理逻辑
        valid_fps, valid_smiles = [], []
        for smi in chunk:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    valid_fps.append(fp)
                    valid_smiles.append(smi)
                except:
                    pass

        # 聚类
        if len(valid_fps) >= 2:
            distance_matrix = []
            for i in range(1, len(valid_fps)):
                sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
                distance_matrix.extend([1 - x for x in sims])
            clusters = Butina.ClusterData(distance_matrix, len(valid_fps), 0.4, True)
            representatives = [valid_smiles[c[0]] for c in clusters]
        else:
            representatives = valid_smiles

        # 保存结果
        output_file = self.output_dir / f"result_{chunk_id}.smi"
        with open(output_file, "w") as f:
            f.write("\n".join(representatives))

        # 更新数据库
        conn = sqlite3.connect(self.status_db)
        conn.execute("UPDATE chunks SET status=? WHERE id=?", ("done", chunk_id))
        conn.commit()
        conn.close()

    def run_parallel(self):
        """生成parallel命令并执行"""
        # 生成任务列表
        task_file = self.output_dir / "parallel_tasks.txt"
        with open(task_file, "w") as f:
            conn = sqlite3.connect(self.status_db)
            for row in conn.execute('SELECT id,file FROM chunks WHERE status="pending"'):
                f.write(f"{row[0]} {row[1]}\n")
            conn.close()

        # 构建parallel命令
        parallel_cmd = (
            f"cat {task_file} | parallel --col-sep ' ' "
            f"'{sys.executable} {__file__} --process_chunk "
            f"--output {self.output_dir} "
            f"--chunk_id {{1}} --chunk_file {{2}}'"
        )

        print(f"请执行以下命令启动并行处理:\n{parallel_cmd}")

    def final_clustering(self, target=100000):
        """最终聚类步骤（与原脚本相同）"""
        all_reps = []
        for f in self.output_dir.glob("result_*.smi"):
            with open(f) as fr:
                all_reps.extend(fr.read().splitlines())

        # 后续聚类逻辑与原脚本保持一致
        # ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--process_chunk", action="store_true")
    parser.add_argument("--chunk_id", type=int)
    parser.add_argument("--chunk_file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.prepare:
        processor = ParallelProcessor("/home/xw3763/project/data/smiles.npy", "./output")
        processor.prepare_chunks()
    elif args.process_chunk:
        processor = ParallelProcessor("/home/xw3763/project/data/smiles.npy", args.output)
        processor.process_single_chunk(args.chunk_file, args.chunk_id)
    elif args.run:
        processor = ParallelProcessor("/home/xw3763/project/data/smiles.npy", "./output")
        processor.run_parallel()
