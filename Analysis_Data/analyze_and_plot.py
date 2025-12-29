import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from io import StringIO

# 读取本地真实 CSV 文件
file_path = r"D:\chenxvsheji\SRTP_MinMax\Path planning-6\experiment_summary.csv"
df = pd.read_csv(file_path)

# 打印前几行看看数据是否读取成功（可选）
print("成功读取数据，前5行如下：")
print(df.head())
# ==========================================
# 2. Wilcoxon 检验函数
# ==========================================
def perform_comparison(df, algo_pair_name, col_hsa, col_ref):
    """
    对比 HSA (col_hsa) 和 参考算法 (col_ref)
    """
    # 1. 过滤：只保留双方都有数据的行
    valid_data = df.dropna(subset=[col_hsa, col_ref]).copy()
    
    if len(valid_data) == 0:
        print(f"[{algo_pair_name}] 没有有效的成对数据，跳过。")
        return None

    # 提取数组
    x = valid_data[col_hsa].values
    y = valid_data[col_ref].values
    n = len(x)

    # 2. 计算 Wins/Ties/Losses (针对最小化问题)
    # Win: HSA < Ref (严格小于)
    # Tie: HSA == Ref
    # Loss: HSA > Ref
    # 注意浮点数比较，可能需要一点容差，这里直接比较
    wins = np.sum(x < y - 1e-9) 
    ties = np.sum(np.abs(x - y) < 1e-9)
    losses = np.sum(x > y + 1e-9)

    # 3. Wilcoxon 检验
    # alternative='less' 表示备择假设是 HSA < Ref
    # 如果数据完全一样（全Tie），wilcoxon会报错或警告
    try:
        # 只有当存在差异时才能做检验
        if np.all(x == y):
            p_val = 1.0
        else:
            res = wilcoxon(x, y, alternative='less')
            p_val = res.pvalue
    except Exception as e:
        p_val = 1.0
        print(f"Wilcoxon Error: {e}")

    print("-" * 60)
    print(f"对比组: {algo_pair_name}")
    print(f"样本量: {n}")
    print(f"Wins: {wins}, Ties: {ties}, Losses: {losses}")
    print(f"P-value: {p_val:.2e}")
    
    return {
        "Pair": algo_pair_name,
        "Wins": wins,
        "Ties": ties,
        "Losses": losses,
        "p_value": p_val
    }

# ==========================================
# 3. 执行对比
# ==========================================
print("正在执行统计分析...")
results = []

# 对比 1: HSA vs Gurobi (主要针对小规模算例)
# 逻辑：只要 Gurobi_Dist 不为空，就进行对比
res_gurobi = perform_comparison(df, "AILS vs. Gurobi", "Hybrid_Dist", "Gurobi_Dist")
if res_gurobi: results.append(res_gurobi)

# 对比 2: HSA vs ILS (Baseline) (主要针对大规模算例)
# 逻辑：只要 Baseline_Dist 不为空，就进行对比
res_ils = perform_comparison(df, "AILS vs. ILS", "Hybrid_Dist", "Baseline_Dist")
if res_ils: results.append(res_ils)

# ==========================================
# 4. 绘图 (归一化 Gap 对比版)
# ==========================================
def plot_results(df):
    # --- 1. 数据预处理：构建动态基准 (Dynamic Baseline) ---
    plot_df = df.copy()
    
    # 确保 Dimension 是数字类型
    plot_df['Dimension'] = pd.to_numeric(plot_df['Dimension'], errors='coerce')
    
    # 排序
    df_sorted = plot_df.sort_values(by='Dimension').reset_index(drop=True)
    x_axis = df_sorted.index + 1
    
    # 核心逻辑：根据 Dimension 选择基准值
    # < 150: 基准是 Gurobi
    # >= 150: 基准是 ILS (Baseline_Dist)
    conditions = [
        (df_sorted['Dimension'] < 150),
        (df_sorted['Dimension'] >= 150)
    ]
    choices = [
        df_sorted['Gurobi_Dist'],
        df_sorted['Baseline_Dist']
    ]
    
    # 创建 "Reference_Value" 列
    df_sorted['Reference_Value'] = np.select(conditions, choices, default=np.nan)
    
    # 计算 HSA 相对于基准的 Gap (%)
    # Gap < 0 (负数) 说明 HSA < Ref (HSA更好)，图像向下
    df_sorted['AILS_Dev_Gap'] = (df_sorted['Hybrid_Dist'] - df_sorted['Reference_Value']) / df_sorted['Reference_Value'] * 100

    # --- 2. 绘图设置 ---
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    except:
        pass

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- 3. 绘制基准线 (y=0) ---
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', label='Baseline (Gurobi / ILS)', zorder=5)

    # --- 4. 绘制 HSA 偏差曲线 ---
    # 只有当有有效数据时才画
    mask = df_sorted['AILS_Dev_Gap'].notna()
    
    ax.plot(x_axis[mask], df_sorted.loc[mask, 'AILS_Dev_Gap'], 
             color='#d62728',       # 红色
             linewidth=1.5, 
             marker='o', 
             markersize=4, 
             label='AILS Deviation',
             zorder=10)

    # --- 5. 区域填充 (增强视觉效果) ---
    # 0以下填充绿色 (Better)，0以上填充红色 (Worse)
    ax.fill_between(x_axis[mask], 0, df_sorted.loc[mask, 'AILS_Dev_Gap'], 
                    where=(df_sorted.loc[mask, 'AILS_Dev_Gap'] < 0),
                    color='green', alpha=0.1, interpolate=True, label='AILS Better')
    
    ax.fill_between(x_axis[mask], 0, df_sorted.loc[mask, 'AILS_Dev_Gap'], 
                    where=(df_sorted.loc[mask, 'AILS_Dev_Gap'] > 0),
                    color='red', alpha=0.1, interpolate=True, label='AILS Worse')
    # --- 6. 分界线与标注 ---
    # 找到 150 分界点的位置
    cutoff_indices = df_sorted.index[df_sorted['Dimension'] >= 150].tolist()
    if cutoff_indices:
        cutoff_idx = cutoff_indices[0] + 1 # +1 对应 x_axis
        
        # 画一条垂直虚线
        ax.axvline(x=cutoff_idx - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        
        # 标注区域文字
        # 左侧区域
        ax.text(cutoff_idx/2, ax.get_ylim()[1]*0.9, 'Small Scale\n(Baseline: Gurobi)', 
                ha='center', va='top', fontsize=12, fontweight='bold', color='#555555', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 右侧区域
        mid_right = cutoff_idx + (len(df_sorted) - cutoff_idx)/2
        ax.text(mid_right, ax.get_ylim()[1]*0.9, 'Large Scale\n(Baseline: ILS)', 
                ha='center', va='top', fontsize=12, fontweight='bold', color='#555555',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # --- 7. 坐标轴处理 ---
    # 使用 Symlog 以适应可能出现的个别大偏差，同时保留0附近的细节
    ax.set_yscale('symlog', linthresh=1.0) 
    
    # 手动设置一些刻度，方便阅读
    ax.set_yticks([-100, -10, -1, 0, 1, 10, 100])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel('Instance Index (Sorted by Dimension)', fontweight='bold')
    ax.set_ylabel('Gap to Baseline (%) \n(Negative is Better)', fontweight='bold')
    
    ax.grid(True, axis='y', which='major', linestyle='--', alpha=0.4)
    sns.despine()

    # 图例
    # 重新组织图例，使其更清晰
    handles, labels = ax.get_legend_handles_labels()
    # 过滤掉填充的图例，或者保留看你喜好
    ax.legend(loc='lower left', frameon=True, framealpha=0.95, ncol=2)

    plt.tight_layout()
    plt.savefig('performance_deviation_plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('performance_deviation_plot.png', dpi=300, bbox_inches='tight')
    print("\n✅ 基准偏差图绘制完成！已保存。")
    plt.show()

# 运行
plot_results(df)