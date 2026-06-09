import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp
import harmonypy
import shutil
import re
from collections import defaultdict
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# 创建工作目录
dirs_to_create = [
    'input',
    'filtered_output',
    'imputed_output',
    'normalized_output',
    'pca_output',
    'harmony_output',
    'clustering_plots',
    'cluster_csvs',
    'top_plots'
]
for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

# 设置 Scanpy 参数
sc.settings.verbosity = 3  # 设置日志详细程度
sc.logging.print_header()
sc.settings.random_seed = 42 # 设置全局随机种子


# 1. 加载数据
adata = sc.read_h5ad(input_path)
print(f"过滤前数据维度: {adata.n_obs} 细胞 × {adata.n_vars} 基因")

# 2. 过滤低表达基因（在少于3个细胞中表达的基因）
sc.pp.filter_genes(adata, min_cells=3)

# 3. 保存过滤后的数据
filtered_output_path = 'filtered_output/final_dataset_filtered.h5ad'
adata.write_h5ad(filtered_output_path)

print(f"过滤完成！已保存为: {filtered_output_path}")
print(f"过滤后数据维度: {adata.n_obs} 细胞 × {adata.n_vars} 基因")


def fill_nan_with_mean(adata):
    """用数据集中所有非NaN值的均值填充NaN。"""
    X = adata.X

    # 如果是稀疏矩阵，先转换为密集矩阵
    if sp.issparse(X):
        X = X.toarray()

    # 计算所有非nan值的均值
    mean_value = np.nanmean(X)
    print(f"计算出的全局均值为: {mean_value:.4f}")

    # 用均值填充nan值
    X_filled = np.where(np.isnan(X), mean_value, X)

    # 更新adata对象
    adata.X = X_filled
    return adata

# 读取上一步过滤后的数据
adata = anndata.read_h5ad(filtered_output_path)

# 填充nan值
adata = fill_nan_with_mean(adata)

# 保存结果
imputed_output_path = 'imputed_output/final_dataset_filled.h5ad'
adata.write(imputed_output_path)
print(f"缺失值填充完成，已保存为: {imputed_output_path}")


# 加载上一步填充后的数据
adata = sc.read_h5ad(imputed_output_path)

# 标准化处理
print("开始标准化...")
sc.pp.normalize_total(adata, target_sum=1e5)

# 对数据进行对数化转换
# sc.pp.log1p(adata)
# print("标准化与对数转换完成。")

# 保存标准化后的数据
normalized_output_path = 'normalized_output/final_dataset_norm.h5ad'
adata.write(normalized_output_path)
print(f"标准化数据已保存: {normalized_output_path}")


def run_pca_on_directory(input_dir, output_dir, n_comps_list=[50], random_state=42):
    """
    对指定文件夹内所有.h5ad文件运行PCA降维。
    先降维至n_comps_list中的最大维度，再从中提取前n个特征保存不同维度的结果。
    """
    os.makedirs(output_dir, exist_ok=True)
    max_n_comps = max(n_comps_list)

    for file in os.listdir(input_dir):
        if file.endswith(".h5ad"):
            print(f"\n正在处理 {file}...")
            adata = sc.read_h5ad(os.path.join(input_dir, file))

            # 执行PCA到最大维度
            sc.tl.pca(
                adata,
                n_comps=max_n_comps,
                svd_solver='arpack',
                random_state=random_state
            )

            base_name = os.path.splitext(file)[0]
            for n_comps in n_comps_list:
                # 创建一个新的AnnData对象，只包含所需的PCA结果
                adata_pca = sc.AnnData(
                    obs=adata.obs.copy(),
                    obsm={'X_pca': adata.obsm['X_pca'][:, :n_comps].copy()},
                    var=adata.var.copy(),
                    uns=adata.uns.copy()
                )
                adata_pca.uns['pca_params'] = {'n_comps': n_comps, 'random_state': random_state}

                output_name = f"{base_name}_pca_ncomps{n_comps}.h5ad"
                adata_pca.write_h5ad(os.path.join(output_dir, output_name))
                print(f"已保存PCA结果 (n_comps={n_comps}) 至 {output_name}")

# 定义参数并运行
pca_params = {
    'n_comps_list': [10, 20, 30, 50],
    'random_state': 42
}

run_pca_on_directory(
    input_dir="normalized_output",
    output_dir="pca_output",
    **pca_params
)


def harmony_batch_correction(input_dir, output_dir, harmony_params, random_seed=42):
    """
    对输入文件夹中的所有.h5ad文件运行Harmony批次校正。
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)  # 固定随机种子

    for file in os.listdir(input_dir):
        if file.endswith(".h5ad"):
            print(f"\n正在处理 {file}...")
            adata = sc.read_h5ad(os.path.join(input_dir, file))
            
            if 'batch' not in adata.obs.columns:
                print(f"警告: 文件 {file} 中缺少 'batch' 列，跳过批次校正。")
                continue
            
            # Harmony批次校正
            sc.external.pp.harmony_integrate(
                adata,
                key='batch', # 指定包含批次信息的列
                basis='X_pca', # 在PCA空间上运行
                **harmony_params,
                random_state=random_seed
            )

            # 将校正后的结果统一命名为 X_pca_harmony
            # adata.obsm['X_pca_corrected'] = adata.obsm['X_pca_harmony'].copy()

            output_name = f"{os.path.splitext(file)[0]}_harmony.h5ad"
            adata.write_h5ad(os.path.join(output_dir, output_name))
            print(f"Harmony校正完成，已保存为: {output_name}")

# 参数设置
harmony_params = {'theta': 2, 'max_iter_harmony': 30}

# 运行
harmony_batch_correction(
    input_dir="pca_output",
    output_dir="harmony_output",
    harmony_params=harmony_params,
    random_seed=42
)


def leiden_clustering_and_eval(
    input_dir,
    csv_dir,
    plot_dir,
    resolution,
    n_neighbors,
    use_rep='X_pca_harmony' # 使用Harmony校正后的表示
):
    """对输入文件运行Leiden聚类，并保存可视化结果和聚类评分"""
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    sc.settings.figdir = plot_dir # 设置图片保存路径

    for file in os.listdir(input_dir):
        if file.endswith(".h5ad"):
            print(f"\n聚类处理: {file} (res={resolution}, k={n_neighbors})")
            adata = sc.read_h5ad(os.path.join(input_dir, file))
            base_name = os.path.splitext(file)[0]

            # 计算邻接图
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=42)

            # Leiden聚类
            sc.tl.leiden(adata, resolution=resolution, key_added='leiden', random_state=42)

            # 计算聚类评分
            X = adata.obsm[use_rep]
            labels = adata.obs['leiden']
            if len(set(labels)) > 1: # 评分至少需要2个簇
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
                davies = davies_bouldin_score(X, labels)
            else:
                silhouette, calinski, davies = np.nan, np.nan, np.nan

            # 保存聚类结果到CSV
            csv_name = f"{base_name}_res{resolution}_k{n_neighbors}_sil{silhouette:.3f}_cal{calinski:.1f}_dav{davies:.3f}.csv"
            pd.DataFrame({'ID': adata.obs.index, 'leiden_cluster': labels}).to_csv(os.path.join(csv_dir, csv_name), index=False)

            # UMAP/t-SNE 可视化
            sc.tl.umap(adata, random_state=42)
            plot_title = f"res={resolution}, k={n_neighbors}\nSil={silhouette:.2f} | Cal={calinski:.0f} | Dav={davies:.2f}"
            sc.pl.umap(adata, color='leiden', title=plot_title, legend_loc='on data', 
                       save=f"_{base_name}_res{resolution}_k{n_neighbors}.png", show=False)
            plt.close() # 关闭图形，防止在notebook中显示
            
# 定义参数网格
param_grid = {
    'resolution': [0.5, 1.0, 2.0, 3.0],
    'n_neighbors': [10, 20, 40]
}

# 遍历所有参数组合并运行
# 注意：这将生成大量文件，根据您的数据大小，可能需要很长时间
for params in product(*param_grid.values()):
    current_params = {'resolution': params[0], 'n_neighbors': params[1]}
    leiden_clustering_and_eval(
        input_dir="harmony_output",
        csv_dir="cluster_csvs",
        plot_dir="clustering_plots",
        **current_params
    )


# 定义参数网格（需要和上一步Leiden聚类中的一致）
res_values = [0.5, 1.0, 2.0, 3.0]
k_values = [10, 20, 40]

# 路径配置
plot_dir = 'clustering_plots'  # 包含所有UMAP图的文件夹
output_base_dir = 'top_plots'   # 输出最佳结果的文件夹

file_groups = defaultdict(list)
# 正则表达式，从文件名中提取参数和分数
pattern = re.compile(r'umap_final_dataset_norm_pca_ncomps(\d+)_harmony_res([\d.]+)_k(\d+).png')
# 从CSV文件名读取更准确的分数
csv_pattern = re.compile(r'final_dataset_norm_pca_ncomps(\d+)_harmony_res([\d.]+)_k(\d+)_sil([\d.\-]+)_cal([\d.\-]+)_dav([\d.\-]+).csv')

# --- 1. 从CSV文件名中收集所有文件的信息和准确分数 ---
all_files_info = defaultdict(list)
for filename in os.listdir('cluster_csvs'):
    match = csv_pattern.match(filename)
    if match:
        ncomps, res, k, sil, cal, dav = match.groups()
        group_key = f"ncomps{ncomps}"
        # 对应的图片文件名
        img_filename = f"umap_final_dataset_norm_pca_ncomps{ncomps}_harmony_res{res}_k{k}.png"
        all_files_info[group_key].append({
            'filename': img_filename,
            'res': float(res),
            'k': int(k),
            'sil': float(sil),
            'cal': float(cal),
            'dav': float(dav)
        })

# --- 2. 为每个分组计算综合分并选出最优 ---
def get_adjacent_params(res, k, res_list, k_list):
    """获取参数网格中的邻居"""
    adjacent = []
    try:
        res_idx = res_list.index(res)
        k_idx = k_list.index(k)
        if res_idx > 0: adjacent.append((res_list[res_idx - 1], k))
        if res_idx < len(res_list) - 1: adjacent.append((res_list[res_idx + 1], k))
        if k_idx > 0: adjacent.append((res, k_list[k_idx - 1]))
        if k_idx < len(k_list) - 1: adjacent.append((res, k_list[k_idx + 1]))
    except ValueError:
        pass # 如果参数不在列表中则忽略
    return adjacent

for group_key, files in all_files_info.items():
    if not files:
        continue

    print(f"\n正在处理分组: {group_key}")

    # 提取指标用于归一化
    sils = np.array([f['sil'] for f in files])
    cals = np.array([f['cal'] for f in files])
    davs = np.array([f['dav'] for f in files])

    # 归一化 (处理分母为0的情况)
    norm_sils = (sils - sils.min()) / (sils.max() - sils.min()) if (sils.max() - sils.min()) != 0 else np.zeros_like(sils)
    norm_cals = (cals - cals.min()) / (cals.max() - cals.min()) if (cals.max() - cals.min()) != 0 else np.zeros_like(cals)
    norm_davs = (davs - davs.min()) / (davs.max() - davs.min()) if (davs.max() - davs.min()) != 0 else np.zeros_like(davs)

    # 计算综合分
    for i, f in enumerate(files):
        f['score'] = norm_sils[i] + norm_cals[i] - norm_davs[i]

    files_sorted = sorted(files, key=lambda x: x['score'], reverse=True)
    
    param_grid_status = {(res, k): {'selected': False, 'blocked': False} for res in res_values for k in k_values}
    selected_files = []

    # 贪心选择
    for f in files_sorted:
        if len(selected_files) >= 5: break
        res, k = f['res'], f['k']
        if not param_grid_status.get((res, k), {}).get('blocked'):
            selected_files.append(f)
            param_grid_status[(res, k)]['selected'] = True
            # 屏蔽邻居
            for adj_res, adj_k in get_adjacent_params(res, k, res_values, k_values):
                if (adj_res, adj_k) in param_grid_status:
                    param_grid_status[(adj_res, adj_k)]['blocked'] = True

    # 复制选出的文件
    group_dir = os.path.join(output_base_dir, group_key)
    os.makedirs(group_dir, exist_ok=True)
    for i, f in enumerate(selected_files, 1):
        src = os.path.join(plot_dir, f['filename'])
        if os.path.exists(src):
            # 在新文件名中加入排名和分数
            dst_filename = f"top{i}_score{f['score']:.3f}_{f['filename']}".replace('umap_','',1)
            dst = os.path.join(group_dir, dst_filename)
            shutil.copy2(src, dst)
            print(f"已复制 Top {i}: {dst_filename}")
        else:
            print(f"警告: 未找到图片文件 {f['filename']}")

print("\n结果筛选完成！")

