#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd

# --- 模块一：初始化 ---
def initialize_population(N, mean=50, std=10):
    """
    初始化 N 对个体，并进行物理索引随机化，确保位置对称性。
    """
    abilities = np.random.normal(mean, std, 2 * N)
    groups = np.array(['A'] * N + ['B'] * N)
    p = np.random.permutation(2 * N)
    return abilities[p], groups[p]

# --- 模块二：单轮更新 (Softmax Selection) ---
def run_single_round(abilities, groups, beta, top_k, delta, epsilon_std, tau):
    """
    基于温度参数 tau 的概率性竞争逻辑。
    """
    N_total = len(abilities)
    num_winners = int(N_total * top_k)
    
    # 1. 计算感知能力 (偏见仅作用于感知层)
    perceived = abilities.copy()
    perceived[groups == 'B'] -= beta
    
    # 2. 数值稳定的 Softmax 概率映射
    scaled = perceived / tau
    scaled -= np.max(scaled) 
    exp_scores = np.exp(scaled)
    probs = exp_scores / np.sum(exp_scores)
    
    # 3. 按概率进行无放回抽样，模拟具有流动性的竞争
    winner_idx = np.random.choice(
        np.arange(N_total),
        size=num_winners,
        replace=False,
        p=probs
    )
    
    # 4. 更新能力：仅获胜者获得增量红利
    if delta > 0:
        noise = np.random.normal(0, epsilon_std, num_winners)
        abilities[winner_idx] += (delta + noise)
    
    # 5. 聚合统计指标
    m_a = np.mean(abilities[groups == 'A'])
    m_b = np.mean(abilities[groups == 'B'])
    a_win_ratio = np.mean(groups[winner_idx] == 'A')
    
    return abilities, {
        'mean_A': m_a, 
        'mean_B': m_b, 
        'gap': m_a - m_b, 
        'A_winner_ratio': a_win_ratio
    }

# --- 模块三：模拟主循环 ---
def run_full_simulation(N, T, beta, top_k, delta, epsilon_std, tau=1.0):
    """
    执行 T 轮迭代，并返回包含所有过程数据的 DataFrame。
    """
    abilities, groups = initialize_population(N)
    history = []
    
    for r in range(T):
        abilities, stats = run_single_round(
            abilities, groups, beta, top_k, delta, epsilon_std, tau
        )
        stats['round'] = r
        history.append(stats)
        
    return pd.DataFrame(history)



# In[16]:


# --- 模块四：M 次重复模拟与参数扫描 ---

def run_multiple_simulations(N, T, beta, top_k, delta, epsilon_std, tau, M=100):
    """
    运行 M 次独立重复实验，提取时序演化的均值与置信区间统计量。
    """
    all_gaps = []
    all_ratios = []
    
    for _ in range(M):
        df = run_full_simulation(N, T, beta, top_k, delta, epsilon_std, tau)
        all_gaps.append(df['gap'].values)
        all_ratios.append(df['A_winner_ratio'].values)
    
    all_gaps = np.array(all_gaps)
    all_ratios = np.array(all_ratios)
    
    # 构造统计 DataFrame
    results = pd.DataFrame({'round': np.arange(T)})
    
    # 计算 Gap 的统计量
    results['gap_mean'] = np.mean(all_gaps, axis=0)
    results['gap_std'] = np.std(all_gaps, axis=0)
    
    # 计算 Ratio 的统计量
    results['ratio_mean'] = np.mean(all_ratios, axis=0)
    results['ratio_std'] = np.std(all_ratios, axis=0)
    
    # 统一计算 95% 置信区间
    error_const = 1.96 / np.sqrt(M)
    results['gap_ci_upper'] = results['gap_mean'] + error_const * results['gap_std']
    results['gap_ci_lower'] = results['gap_mean'] - error_const * results['gap_std']
    results['ratio_ci_upper'] = results['ratio_mean'] + error_const * results['ratio_std']
    results['ratio_ci_lower'] = results['ratio_mean'] - error_const * results['ratio_std']
    
    return results

def beta_scan(N, T, beta_values, top_k, delta, epsilon_std, tau, M=100):
    """
    扫描不同 beta，记录系统最终进入稳态后的指标。
    """
    scan_data = []
    
    for b in beta_values:
        print(f"Scanning Beta: {b:.2f}...")
        final_gaps = []
        steady_state_ratios = []
        
        for _ in range(M):
            df = run_full_simulation(N, T, b, top_k, delta, epsilon_std, tau)
            # 记录最后一轮的差距作为最终演化结果
            final_gaps.append(df['gap'].iloc[-1])
            # 【改进】取最后 10 轮的均值，捕捉系统稳态的机会俘获水平
            steady_state_ratios.append(df['A_winner_ratio'].iloc[-10:].mean())
            
        scan_data.append({
            'beta': b,
            'final_gap_mean': np.mean(final_gaps),
            'final_gap_std': np.std(final_gaps),
            'steady_A_ratio': np.mean(steady_state_ratios),
            'steady_A_ratio_std': np.std(steady_state_ratios) # 记录稳态波动的标准差
        })
        
    return pd.DataFrame(scan_data)
def T_scan(N, T_values, beta, top_k, delta, epsilon_std, tau, M=100):
    """
    扫描不同 T (迭代轮数)，观察不平等随时间跨度的演化极限。
    """
    scan_data = []
    
    for t in T_values:
        print(f"Scanning T: {t}...")
        final_gaps = []
        
        for _ in range(M):
            df = run_full_simulation(N, t, beta, top_k, delta, epsilon_std, tau)
            final_gaps.append(df['gap'].iloc[-1])
            
        scan_data.append({
            'T': t,
            'final_gap_mean': np.mean(final_gaps),
            'final_gap_std': np.std(final_gaps)
        })
        
    return pd.DataFrame(scan_data)
def joint_scan(N, T, beta, top_k_values, tau_values, delta, epsilon_std, M=30):
    """
    联合扫描 top_k 和 tau，记录稳态 gap 和 winner ratio。
    M 默认用 30，因为是二维扫描，计算量较大。
    返回适合画热图的宽格式 DataFrame。
    """
    # 用字典存两个指标的结果矩阵
    gap_matrix = np.zeros((len(tau_values), len(top_k_values)))
    ratio_matrix = np.zeros((len(tau_values), len(top_k_values)))

    for i, tau in enumerate(tau_values):
        for j, tk in enumerate(top_k_values):
            print(f"Scanning top_k={tk:.2f}, tau={tau:.2f}...")
            final_gaps = []
            final_ratios = []

            for _ in range(M):
                df = run_full_simulation(N, T, beta, tk, delta, epsilon_std, tau)
                final_gaps.append(df['gap'].iloc[-1])
                final_ratios.append(df['A_winner_ratio'].iloc[-10:].mean())

            gap_matrix[i, j] = np.mean(final_gaps)
            ratio_matrix[i, j] = np.mean(final_ratios)

    # 转成 DataFrame，行是 tau，列是 top_k
    gap_df = pd.DataFrame(gap_matrix, 
                          index=[f"{t:.1f}" for t in tau_values],
                          columns=[f"{k:.2f}" for k in top_k_values])
    ratio_df = pd.DataFrame(ratio_matrix,
                            index=[f"{t:.1f}" for t in tau_values],
                            columns=[f"{k:.2f}" for k in top_k_values])

    return gap_df, ratio_df


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# --- 模块五：可视化 ---

def plot_final_research_split(scan_df, title_prefix="Elite Competition Analysis"):
    """
    分离绘制：图1展示能力差距演化，图2展示机会俘获比例。
    """
    sns.set_theme(style="ticks")
    
    # --- 图 1: Ability Gap ---
    plt.figure(figsize=(8, 5))
    plt.errorbar(scan_df['beta'], scan_df['final_gap_mean'], 
                 yerr=1.96 * scan_df['final_gap_std'] / np.sqrt(M), 
                 fmt='o', color='#D7191C', markersize=6, capsize=3, alpha=0.8)
    plt.title(f"{title_prefix}: Ability Gap Growth", fontsize=13, fontweight='bold')
    plt.xlabel('Bias Intensity (Beta)', fontsize=11)
    plt.ylabel('Final Ability Gap (A - B)', fontsize=11)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

    # --- 图 2: Winner Ratio ---
    plt.figure(figsize=(8, 5))
    plt.plot(scan_df['beta'], scan_df['steady_A_ratio'], 
             's-', color='#2C7BB6', markersize=5, linewidth=1.5, alpha=0.7)
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.4)
    plt.title(f"{title_prefix}: Opportunity Capture", fontsize=13, fontweight='bold')
    plt.xlabel('Bias Intensity (Beta)', fontsize=11)
    plt.ylabel('A-Group Winner Ratio', fontsize=11)
    plt.ylim(0.4, 1.05)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()


def plot_time_series(results_df, beta, title_prefix="Bias Amplification Dynamics"):
    """
    绘制 run_multiple_simulations 的时序输出。
    results_df: run_multiple_simulations 返回的 DataFrame
    beta: 当前使用的偏见强度（用于标题）
    """
    sns.set_theme(style="ticks")
    rounds = results_df['round']

    # --- 图1：Ability Gap 时序 ---
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, results_df['gap_mean'], 
             color='#D7191C', linewidth=2, label='Mean Gap')
    plt.fill_between(rounds,
                     results_df['gap_ci_lower'],
                     results_df['gap_ci_upper'],
                     color='#D7191C', alpha=0.2, label='95% CI')
    plt.title(f"{title_prefix} (β={beta}): Ability Gap over Time", 
              fontsize=13, fontweight='bold')
    plt.xlabel('Round', fontsize=11)
    plt.ylabel('Ability Gap (A - B)', fontsize=11)
    plt.legend(fontsize=10)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- 图2：Winner Ratio 时序 ---
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, results_df['ratio_mean'],
             color='#2C7BB6', linewidth=2, label='Mean A-Winner Ratio')
    plt.fill_between(rounds,
                     results_df['ratio_ci_lower'],
                     results_df['ratio_ci_upper'],
                     color='#2C7BB6', alpha=0.2, label='95% CI')
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.4, label='Baseline (0.5)')
    plt.title(f"{title_prefix} (β={beta}): A-Group Win Ratio over Time",
              fontsize=13, fontweight='bold')
    plt.xlabel('Round', fontsize=11)
    plt.ylabel('A-Group Winner Ratio', fontsize=11)
    plt.legend(fontsize=10)
    plt.ylim(0.4, 1.0)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_T_scan(T_scan_df, title_prefix="Bias Amplification"):
    """
    绘制 T_scan 的结果：最终 gap 随迭代轮数 T 的变化。
    T_scan_df: T_scan() 返回的 DataFrame，包含列 T / final_gap_mean / final_gap_std
    """
    sns.set_theme(style="ticks")
    plt.figure(figsize=(8, 5))

    # 均值线
    plt.plot(T_scan_df['T'], T_scan_df['final_gap_mean'],
             'o-', color='#1A9641', linewidth=2, markersize=5, label='Mean Final Gap')

    # 95% 置信区间阴影
    error = 1.96 * T_scan_df['final_gap_std'] / np.sqrt(M)
    plt.fill_between(T_scan_df['T'],
                     T_scan_df['final_gap_mean'] - error,
                     T_scan_df['final_gap_mean'] + error,
                     color='#1A9641', alpha=0.2, label='95% CI')

    plt.title(f"{title_prefix}: Ability Gap vs Time Horizon (T)",
              fontsize=13, fontweight='bold')
    plt.xlabel('Number of Rounds (T)', fontsize=11)
    plt.ylabel('Final Ability Gap (A - B)', fontsize=11)
    plt.legend(fontsize=10)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def extract_slopes(beta_values, N, T, top_k, delta, epsilon_std, tau, M=100):
    """
    对每个 beta 值，运行 M 次模拟，对每次模拟的 gap(t) 时序做线性回归，
    提取斜率（放大速率），汇报均值和置信区间。
    
    返回 DataFrame，包含列：beta / slope_mean / slope_std / r2_mean
    """
    results = []

    for b in beta_values:
        print(f"Extracting slope for beta={b:.2f}...")
        slopes = []
        r2s = []

        for _ in range(M):
            df = run_full_simulation(N, T, b, top_k, delta, epsilon_std, tau)
            
            # 对这一次模拟的 gap 时序做线性回归
            # stats.linregress 返回：slope, intercept, r_value, p_value, std_err
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['round'],  # x：轮次
                df['gap']     # y：当轮 gap 值
            )
            slopes.append(slope)
            r2s.append(r_value ** 2)  # R² 衡量线性拟合质量

        results.append({
            'beta': b,
            'slope_mean': np.mean(slopes),
            'slope_std': np.std(slopes),
            'slope_ci_upper': np.mean(slopes) + 1.96 * np.std(slopes) / np.sqrt(M),
            'slope_ci_lower': np.mean(slopes) - 1.96 * np.std(slopes) / np.sqrt(M),
            'r2_mean': np.mean(r2s)   # 平均 R²，用来验证线性假设
        })

    return pd.DataFrame(results)
def plot_slopes(slope_df, title_prefix="Bias Amplification"):
    """
    绘制 slope vs beta 图，展示偏见强度如何决定放大速率。
    slope_df: extract_slopes() 返回的 DataFrame
    """
    sns.set_theme(style="ticks")
    plt.figure(figsize=(8, 5))

    plt.errorbar(slope_df['beta'], slope_df['slope_mean'],
                 yerr=1.96 * slope_df['slope_std'] / np.sqrt(M),
                 fmt='o-', color='#756BB1', linewidth=2,
                 markersize=6, capsize=3, alpha=0.85, label='Amplification Slope')

    plt.title(f"{title_prefix}: Bias Intensity vs Amplification Rate",
              fontsize=13, fontweight='bold')
    plt.xlabel('Bias Intensity (β)', fontsize=11)
    plt.ylabel('Gap Growth Rate (slope)', fontsize=11)
    plt.legend(fontsize=10)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 顺便打印平均 R²，确认线性拟合质量
    print(f"Mean R² across all beta values: {slope_df['r2_mean'].mean():.4f}")
    
def plot_joint_heatmap(gap_df, ratio_df, beta, title_prefix="Institutional Sensitivity"):
    """
    绘制 top_k × tau 联合扫描热图。
    gap_df / ratio_df: joint_scan() 返回的两个 DataFrame
    行索引是 tau，列索引是 top_k
    """
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 图1：Ability Gap 热图 ---
    sns.heatmap(gap_df,
                ax=axes[0],
                cmap='Reds',
                annot=True,          # 在每个格子里显示数值
                fmt='.1f',           # 数值保留一位小数
                linewidths=0.5,
                cbar_kws={'label': 'Final Ability Gap'})

    axes[0].set_title(f"{title_prefix} (β={beta}): Ability Gap",
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Competition Intensity (top_k)', fontsize=10)
    axes[0].set_ylabel('Selection Sharpness (tau)', fontsize=10)

    # --- 图2：Winner Ratio 热图 ---
    sns.heatmap(ratio_df,
                ax=axes[1],
                cmap='Blues',
                annot=True,
                fmt='.2f',
                linewidths=0.5,
                vmin=0.5,            # 基准线从0.5开始，让颜色对比更清晰
                cbar_kws={'label': 'A-Group Winner Ratio'})

    axes[1].set_title(f"{title_prefix} (β={beta}): Opportunity Capture",
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Competition Intensity (top_k)', fontsize=10)
    axes[1].set_ylabel('Selection Sharpness (tau)', fontsize=10)

    plt.suptitle(f"Joint Sensitivity: top_k × tau  |  β={beta}",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# In[20]:


# --- 主执行入口 ---
if __name__ == "__main__" or True:  
    # 注：在 Jupyter 中 __name__ 不是 "__main__"，加 "or True" 确保可以直接运行
    
    # ============================================================
    # 全局参数设定
    # ============================================================
    N          = 200    # 每组人数（A/B各200，共400人）
    T          = 100    # 模拟轮数
    beta       = 5      # 基准偏见强度
    top_k      = 0.2    # 每轮获胜比例（前20%获得机会）
    delta      = 1.0    # 获胜者每轮能力增量
    epsilon_std= 0.5    # 增量噪声标准差
    tau        = 1.0    # Softmax 温度参数
    M          = 100    # Monte Carlo 重复次数
    
    # ============================================================
    # 分析一：时序演化动态（单一 beta 下的过程）
    # ============================================================
    print("=" * 50)
    print("Analysis 1: Time Series Dynamics")
    print("=" * 50)
    
    ts_results = run_multiple_simulations(
        N=N, T=T, beta=beta, top_k=top_k,
        delta=delta, epsilon_std=epsilon_std, tau=tau, M=M
    )
    plot_time_series(ts_results, beta=beta)
    
    # ============================================================
    # 分析二：T-scan，验证 gap ∝ T 的线性关系
    # ============================================================
    print("=" * 50)
    print("Analysis 2: T-Scan (Gap vs Time Horizon)")
    print("=" * 50)
    
    T_scan_results = T_scan(
        N=N, T_values=[20, 50, 100, 150, 200],
        beta=beta, top_k=top_k,
        delta=delta, epsilon_std=epsilon_std, tau=tau, M=M
    )
    plot_T_scan(T_scan_results)
    
    # ============================================================
    # 分析三：beta-scan，验证 gap 随偏见强度的放大趋势
    # ============================================================
    print("=" * 50)
    print("Analysis 3: Beta-Scan (Gap vs Bias Intensity)")
    print("=" * 50)
    
    beta_values = [0, 2, 4, 6, 8, 10]
    
    beta_scan_results = beta_scan(
        N=N, T=T, beta_values=beta_values,
        top_k=top_k, delta=delta,
        epsilon_std=epsilon_std, tau=tau, M=M
    )
    plot_final_research_split(beta_scan_results)
    
    # ============================================================
    # 分析四：slope 提取，验证 β 决定放大速率
    # ============================================================
    print("=" * 50)
    print("Analysis 4: Slope Extraction (Amplification Rate vs Beta)")
    print("=" * 50)
    
    slope_results = extract_slopes(
        beta_values=beta_values,
        N=N, T=T, top_k=top_k,
        delta=delta, epsilon_std=epsilon_std, tau=tau, M=M
    )
    plot_slopes(slope_results)
    
    # ============================================================
    # 分析五：top_k × tau 联合敏感性热图
    # ============================================================
    print("=" * 50)
    print("Analysis 5: Joint Sensitivity (top_k x tau)")
    print("=" * 50)
    
    gap_df, ratio_df = joint_scan(
        N=N, T=T, beta=beta,
        top_k_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        tau_values=[0.5, 1.0, 2.0, 5.0, 10.0],
        delta=delta, epsilon_std=epsilon_std, M=30
    )
    plot_joint_heatmap(gap_df, ratio_df, beta=beta)
    
    print("=" * 50)
    print("All analyses complete.")
    print("=" * 50)


# In[ ]:




