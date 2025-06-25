import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import os


# ========== 自定义国家参数 ==========
countries = [
    {
        'name': 'Namibia',
        'data_path': '../Data/Compartment_NA.csv',
        'start_date': datetime(2020, 5, 15),
        'N': 2540916,
        'beta_n': 0.6075,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
    },
    {
        'name': 'Singapore',
        'data_path': '../Data/Compartment_SG.csv',
        'start_date': datetime(2020, 1, 22),
        'N': 5850343,
        'beta_n': 0.4429,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
    },
    {
        'name': 'Germany',
        'data_path': '../Data/Compartment_DE.csv',
        'start_date': datetime(2020, 1, 22),
        'N': 83019213,
        'beta_n': 0.5015,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
    },
    {
        'name': 'Brazil',
        'data_path': '../Data/Compartment_BR.csv',
        'start_date': datetime(2020, 2, 26),
        'N': 212559409,
        'beta_n': 0.4944,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
    },
    {
        'name': 'Austrialia',
        'data_path': '../Data/Compartment_AU.csv',
        'start_date': datetime(2021, 10, 4),
        'N': 25499881,
        'beta_n': 0.4444,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
    },
    {
        'name': 'America',
        'data_path': '../Data/Compartment_US.csv',
        'start_date': datetime(2020, 1, 18),
        'N': 331002647,
        'beta_n': 0.4944,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
    }
]
countries = sorted(countries, key=lambda x: x['name'])

for i, country in enumerate(countries):

    # 读取数据
    data = pd.read_csv(country['data_path'], header=0).values
    N = country['N']
    start_date = country['start_date']
    beta_n = country['beta_n']
    beta_a = country['beta_a']
    sigma = country['sigma']
    gamma = country['gamma']
    S, E, I, R, p_values, I_new_nn, I_cum_nn, I_new_real, I_cum_real, f = data.T

    time = np.array([start_date + timedelta(days=int(i)) for i in range(len(S))])

    # 时间范围
    tspan = np.arange(len(S))

    # 初始条件
    S0 = S[0]
    E0 = E[0]
    I0 = I[0]
    R0 = 1 - S0 - E0 - I0
    initial_conditions = [S0 * N, E0 * N, I0 * N, R0 * N]


    # p 函数
    def p(t):
        idx = min(int(round(t)), len(p_values) - 1)
        return p_values[idx]


    # SEIR模型的微分方程
    def SEIR(y, t, beta_n, beta_a, sigma, gamma, p):
        S, E, I, R = y
        dS = -(p(t) * beta_n + (1 - p(t)) * beta_a) * S * I / N
        dE = (p(t) * beta_n + (1 - p(t)) * beta_a) * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]


    sol = odeint(SEIR, initial_conditions, tspan, args=(beta_n, beta_a, sigma, gamma, p))
    I_new_ODE = gamma * sol[:, 2]

    # 绘图
    # fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.32))
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 2.16))

    # 添加整体标题
    # fig.suptitle(f"{country['name']}",
    #              fontsize=10,
    #              y=0.9,
    #              weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # 设置轴字体颜色
    # font_color_1 = '#1F77B4'
    # font_color_2 = '#D62728'
    font_color_1 = '#000000'
    font_color_2 = '#000000'

    # 第一个子图
    ax1 = axes[0]
    ax1.bar(time, I_new_real, color='#7FABD1', edgecolor='#52AADC', alpha=0.6, width=0.3, label='Real')

    ax1.plot(time, I_new_nn, linewidth=2, label='Neural Network', linestyle='-', color='r')
    # ax1.plot(time, I_new_ODE, color='#073E7F', linewidth=2, linestyle='dashed', label='ODE solution')
    max_val = max(max(I_new_real), max(I_new_ODE))
    exponent = int(np.floor(np.log10(max_val))) if max_val > 0 else 0


    # 自定义y轴刻度格式
    def format_y_ticks(x, pos):
        scaled_x = x / (10 ** exponent) if exponent != 0 else x
        return f"{scaled_x:.1f}"


    ax1.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    ax1.tick_params(axis='y', labelsize=10)
    ax1.set_ylabel('Daily confirmed cases', fontsize=12, color=font_color_1)
    # 添加科学计数法标记
    if exponent != 0:
        ax1.text(0.05, 1.02, f'$\\times 10^{{{exponent}}}$', transform=ax1.transAxes,
                ha='right', va='bottom', fontsize=10, color='black')

    start_date = time[0]
    end_date = time[-1]

    # 计算4个等分点（共5个刻度，包括首尾）
    if len(time) > 1:
        date_range = end_date - start_date
        tick_dates = [
            start_date,
            start_date + date_range / 4,
            start_date + 2 * date_range / 4,
            start_date + 3 * date_range / 4,
            end_date
        ]
        ax1.set_xticks(tick_dates)
    else:
        ax1.set_xticks([start_date])

    # 设置日期格式为YYYY.MM
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
    ax1.tick_params(axis='x', labelsize=10)
    plt.xticks(rotation=0, ha='center', fontsize=10)  # 水平居中显示
    # 设置x轴范围
    ax1.set_xlim([start_date, end_date])



    ax1.tick_params(axis='y', labelcolor=font_color_1)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time, p_values, color='#79438E', linewidth=2, label='p')
    ax1_twin.set_ylabel('Unaltered ratio p', fontsize=12, color='#79438E')
    ax1_twin.tick_params(axis='y', labelcolor='#79438E', labelsize=10)
    ax1_twin.legend().set_visible(False)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()

    if i==0:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=9, frameon=False)

    # 第二个子图
    ax2 = axes[1]
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    change_indices = np.where(np.diff(np.sign(f)))[0]
    start_idx = 0
    # 初始化句柄
    h1, h2 = None, None

    for end_idx in change_indices:
        if f[start_idx] < 0:
            if h1 is None:
                h1, = ax2.plot(time[start_idx:end_idx + 2], f[start_idx:end_idx + 2], color='#214EA7', linewidth=2)
            else:
                ax2.plot(time[start_idx:end_idx + 2], f[start_idx:end_idx + 2], color='#214EA7', linewidth=2)
        else:  # f(t) > 0
            if h2 is None:
                h2, = ax2.plot(time[start_idx:end_idx + 2], f[start_idx:end_idx + 2], color='#BF1D2D', linewidth=2)
            else:
                ax2.plot(time[start_idx:end_idx + 2], f[start_idx:end_idx + 2], color='#BF1D2D', linewidth=2)
        start_idx = end_idx + 1

    if f[start_idx] < 0:
        if h1 is None:
            h1, = ax2.plot(time[start_idx:], f[start_idx:], color='#214EA7', linewidth=2)
        else:
            ax2.plot(time[start_idx:], f[start_idx:], color='#214EA7', linewidth=2)
    else:
        if h2 is None:
            h2, = ax2.plot(time[start_idx:], f[start_idx:], color='#BF1D2D', linewidth=2)
        else:
            ax2.plot(time[start_idx:], f[start_idx:], color='#BF1D2D', linewidth=2)

    ax2.axhline(0, color='#909090', linewidth=2, linestyle='--')
    ax2.set_ylabel('Regulatory function f', fontsize=12, color=font_color_2)

    # 计算4个等分点（共5个刻度，包括首尾）
    if len(time) > 1:
        date_range = end_date - start_date
        tick_dates = [
            start_date,
            start_date + date_range / 4,
            start_date + 2 * date_range / 4,
            start_date + 3 * date_range / 4,
            end_date
        ]
        ax2.set_xticks(tick_dates)
    else:
        ax2.set_xticks([start_date])

    # 设置日期格式为YYYY.MM
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
    plt.xticks(rotation=0, ha='center')  # 水平居中显示
    # 设置x轴范围
    ax2.set_xlim([start_date, end_date])

    ax2.tick_params(axis='y', labelsize=10, labelcolor=font_color_2)

    if h1 and h2 and i == 0:
        ax2.legend([h1, h2], ['f(t) < 0', 'f(t) > 0'], loc='lower left', fontsize=9, frameon=False)


    # 调整布局
    fig.align_ylabels(axes)
    plt.tight_layout()

    # 保存图片，文件名使用国家名
    country_name = country['name'].replace(' ', '_')
    output_folder = "../Fig"
    output_path = os.path.join(output_folder, f"{country_name}.tiff")
    plt.savefig(output_path,
                dpi=600,
                format='tiff',
                bbox_inches='tight',  # 去除白边
                pil_kwargs={"compression": "tiff_lzw"}
                )
    plt.close()
