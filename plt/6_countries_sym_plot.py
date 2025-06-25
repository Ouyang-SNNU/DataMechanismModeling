import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import os

# ========== country parameters ==========
countries = [
    {
        'name': 'Namibia',
        'data_path': '../Data/Compartment_NA.csv',
        'data_range': (70, 270),
        'start_date': datetime(2020, 7, 24),
        'N': 2540916,
        'beta_n': 0.6075,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
        'dp_func': lambda y: 11.499 * y[4] * (1 - y[4]) * (1 - 24.3778 * y[0] / 2540916 * (y[2] / 2540916 + 0.0442))
    },
    {
        'name': 'Singapore',
        'data_path': '../Data/Compartment_SG.csv',
        'data_range': (680, 800),
        'start_date': datetime(2021, 12, 2),
        'N': 5850343,
        'beta_n': 0.4429,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
        'dp_func': lambda y: 0.147234309166763 * y[4] * (1 - y[4]) * (1-1.979541009711247e+02*y[2]/5850343)
    },
    {
        'name': 'Germany',
        'data_path': '../Data/Compartment_DE.csv',
        'data_range': (1, 120),
        'start_date': datetime(2021, 1, 22),
        'N': 83019213,
        'beta_n': 0.5015,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
        'dp_func': lambda y: 0.0910*y[4] * (1 - y[4]) * (1-11231*y[2]/83019213)
    },
    {
        'name': 'Brazil',
        'data_path': '../Data/Compartment_BR.csv',
        'data_range': (720, 820),
        'start_date': datetime(2020, 2, 15),
        'N': 212559409,
        'beta_n': 0.4944,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
        'dp_func': lambda y: y[4] * (1 - y[4]) * (0.1567 * (1 - 792.3 * y[0] / 212559409 * (y[2] / 212559409 + 3.663e-6)))
    },
    {
        'name': 'Austrialia',
        'data_path': '../Data/Compartment_AU.csv',
        'data_range': (30, 130),
        'start_date': datetime(2021, 11, 3),
        'N': 25499881,
        'beta_n': 0.4444,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
        'dp_func': lambda y: 0.2563 * y[4] * (1 - y[4]) * (1 - 222.5 * y[2] / 25499881)
    },
    {
        'name': 'America',
        'data_path': '../Data/Compartment_US.csv',
        'start_date': datetime(2020, 10, 14),
        'data_range': (270, 440),
        'N': 331002647,
        'beta_n': 0.4944,
        'beta_a': 0.1,
        'sigma': 0.2,
        'gamma': 0.162 + 4.816e-3,
        'dp_func': lambda y: 0.0099 * y[4] * (1 - y[4]) * (1 - 769.6 * y[2] / 331002647)
    }
]

# Bootstrap
def bootstrap_simulation(country, y0, t_span, t_eval, num_bootstrap=500, noise_std=0.05):
    I_new_all = []

    for _ in range(num_bootstrap):
        beta_n_sample = np.random.normal(country['beta_n'], country['beta_n'] * noise_std)
        beta_a_sample = np.random.normal(country['beta_a'], country['beta_a'] * noise_std)
        def SEIR_bootstrap(t, y):
            beta_eff = y[4] * beta_n_sample + (1 - y[4]) * beta_a_sample
            dSdt = -beta_eff * y[0] * y[2] / country['N']
            dEdt = beta_eff * y[0] * y[2] / country['N'] - country['sigma'] * y[1]
            dIdt = country['sigma'] * y[1] - country['gamma'] * y[2]
            dRdt = country['gamma'] * y[2]
            dpdt = country['dp_func'](y)
            return [dSdt, dEdt, dIdt, dRdt, dpdt]

        sol_bt = solve_ivp(
            SEIR_bootstrap,
            t_span,
            y0,
            t_eval=t_eval,
            method='DOP853',
            rtol=1e-6,
            atol=1e-9,
        )

        I_new_bt = country['gamma'] * sol_bt.y[2]
        I_new_all.append(I_new_bt)

    return np.array(I_new_all)


countries = sorted(countries, key=lambda x: x['name'])
output_folder = "../Fig"
os.makedirs(output_folder, exist_ok=True)

for i, country in enumerate(countries):
    plt.figure(figsize=(4, 2.5), facecolor='white')


    data = pd.read_csv(country['data_path'])
    start_idx, end_idx = country['data_range']
    data_slice = data.iloc[start_idx:end_idx]

    S = data_slice['S'].values
    E = data_slice['E'].values
    I = data_slice['I'].values
    R = data_slice['R'].values
    p_values = data_slice['p'].values
    I_new_real = data_slice['I_new_real'].values


    S0, E0, I0 = S[0], E[0], I[0]
    R0 = 1 - S0 - E0 - I0
    y0 = [S0 * country['N'], E0 * country['N'], I0 * country['N'], R0 * country['N'], p_values[0]]


    time_dates = [country['start_date'] + timedelta(days=int(d)) for d in range(len(S))]
    t_span = (0, len(S) - 1)
    t_eval = np.linspace(0, len(S) - 1, len(S))


    def SEIR(t, y):
        beta_eff = y[4] * country['beta_n'] + (1 - y[4]) * country['beta_a']
        dSdt = -beta_eff * y[0] * y[2] / country['N']
        dEdt = beta_eff * y[0] * y[2] / country['N'] - country['sigma'] * y[1]
        dIdt = country['sigma'] * y[1] - country['gamma'] * y[2]
        dRdt = country['gamma'] * y[2]
        dpdt = country['dp_func'](y)
        return [dSdt, dEdt, dIdt, dRdt, dpdt]


    sol = solve_ivp(
        SEIR,
        t_span,
        y0,
        t_eval=t_eval,
        method='DOP853',
        rtol=1e-6,
        atol=1e-9,
    )
    I_new_ODE = country['gamma'] * sol.y[2]


    I_new_samples = bootstrap_simulation(country, y0, t_span, t_eval,
                                         num_bootstrap=500, noise_std=0.05)
    lower = np.percentile(I_new_samples, 2.5, axis=0)
    upper = np.percentile(I_new_samples, 97.5, axis=0)


    ax = plt.gca()
    ax.plot(time_dates, I_new_ODE, c='#0072BD', linewidth=3, label='ODE Solution', linestyle='-', zorder=2)
    ax.fill_between(time_dates, lower, upper, color='#3A97E3', alpha=0.3, label='95% CI', edgecolor='none', zorder=1)
    ax.scatter(time_dates, I_new_real, s=25, c='#D95319', alpha=0.5, label='Real Data', edgecolor='none', zorder=3)


    ax.set_ylabel('Daily Cases', fontsize=14)
    max_val = max(max(I_new_real), max(I_new_ODE))
    exponent = int(np.floor(np.log10(max_val))) if max_val > 0 else 0


    def format_y_ticks(x, pos):
        scaled_x = x / (10 ** exponent) if exponent != 0 else x
        return f"{scaled_x:.1f}"


    ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    if exponent != 0:
        ax.text(0.1, 1.02, f'$\\times 10^{{{exponent}}}$',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, color='black')


    ax.set_xlim([time_dates[0], time_dates[-1]])
    num_ticks = 4
    if len(time_dates) >= num_ticks:
        tick_indices = np.linspace(0, len(time_dates) - 1, num_ticks, dtype=int)
        xticks = [time_dates[i] for i in tick_indices]
    else:
        xticks = time_dates

    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='right')


    if i == 4:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper left', frameon=False)

    plt.tight_layout()


    country_name = country['name'].replace(' ', '_')
    output_path = os.path.join(output_folder, f"{country_name}_Sym.tiff")
    plt.savefig(
        output_path,
        dpi=600,
        format='tiff',
        bbox_inches='tight',
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()
