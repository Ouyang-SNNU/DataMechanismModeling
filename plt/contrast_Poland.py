import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from holoviews.plotting.bokeh.styles import font_size
from scipy.integrate import odeint
from datetime import datetime, timedelta
import seaborn as sns

# Read data
data = pd.read_csv('../Data/Compartment_PL.csv').values
N = 37972812
beta_a = 0.1
beta_n = 0.3015
sigma = 0.2
gamma = 0.162 + 4.816e-3
start_date = datetime(2020, 2, 19)


data = data[350:430, :]
S = data[:, 0]
E = data[:, 1]
I = data[:, 2]
R = data[:, 3]
p_values = data[:, 4]
I_new_nn = data[:, 5]
I_cum_nn = data[:, 6]
I_new_real = data[:, 7]
I_cum_real = data[:, 8]
f = data[:, 9]

time = [start_date + timedelta(days=i) for i in range(len(S))]
tspan = np.arange(1, len(S) + 1)
len_l2=len(S)

S0 = S[0]
E0 = E[0]
I0 = I[0]
R0 = 1 - S0 - E0 - I0
p0 = p_values[0]
initial_conditions = [S0 * N, E0 * N, I0 * N, R0 * N, p0]

multipliers = [0.3725/0.2220, 1.8, 2]


# Define the SEIRp model
def SEIRp1(y, t):
    dSdt = - (y[4] * beta_n + (1 - y[4]) * beta_a) * y[0] * y[2] / N
    dEdt = (y[4] * beta_n + (1 - y[4]) * beta_a) * y[0] * y[2] / N - sigma * y[1]
    dIdt = sigma * y[1] - gamma * y[2]
    dRdt = gamma * y[2]
    dpdt = 0.2220 * y[4] * (1 - y[4]) * (1-(y[2] / N * 505.3))
    return [dSdt, dEdt, dIdt, dRdt, dpdt]

start_index=10
# Solve ODE
sol1 = odeint(SEIRp1, initial_conditions, tspan)
I_new_ODE_1 = gamma * sol1[:, 2]
initial_conditions_2 = sol1[start_index-1, :]
len_l3= len(S)-start_index+1
I_new_ODE_2 = np.zeros((len_l3, 3))


for i, mult in enumerate(multipliers):
    def SEIRp2(y, t):
        dSdt = - (y[4] * beta_n + (1 - y[4]) * beta_a) * y[0] * y[2] / N
        dEdt = (y[4] * beta_n + (1 - y[4]) * beta_a) * y[0] * y[2] / N - sigma * y[1]
        dIdt = sigma * y[1] - gamma * y[2]
        dRdt = gamma * y[2]
        # dpdt = 0.2220 * y[4] * (1 - y[4]) * (1 - (y[2] / N * 505.3))
        dpdt = mult * 0.2220 * y[4] * (1 - y[4]) * (1-(y[2] / N * 505.3))
        return [dSdt, dEdt, dIdt, dRdt, dpdt]


    sol2 = odeint(SEIRp2, initial_conditions_2, np.arange(0, len_l3))
    I_new_ODE_2[:, i] = gamma * sol2[:, 2]

# Plot results
plt.figure(figsize=(10, 4.5))
l1 = plt.scatter(time, I_new_real, s=80, c='#FF770F', alpha=0.5)
palette = sns.color_palette("colorblind")

# Plot curves
l2, = plt.plot(time[:len_l2], I_new_ODE_1[:len_l2], color='#012696', linewidth=3, linestyle='-')
l3, = plt.plot(time[start_index:], I_new_ODE_2[1:, 0], color='#01847F', linewidth=3, linestyle='-')
plt.axvline(x=time[start_index], color='#606060', linestyle='--', linewidth=3)

# Find peaks
peak_idx_l2 = np.argmax(I_new_ODE_1[:])
peak_val_l2 = I_new_ODE_1[peak_idx_l2]
peak_date_l2 = time[peak_idx_l2]

peak_idx_l3 = np.argmax(I_new_ODE_2[:, 0])
peak_val_l3 = I_new_ODE_2[:, 0][peak_idx_l3]
peak_date_l3 = time[start_index-1 + peak_idx_l3]

# Calculate cumulative cases from change date to peak
change_idx = 10

# l2
cum_l2 = np.sum(I_new_ODE_1[start_index-1:peak_idx_l2+1])

# l3
cum_l3 = np.sum(I_new_ODE_2[:peak_idx_l3+1, 0])

cum_diff = cum_l2 - cum_l3


# Mark peaks with circles
plt.scatter([peak_date_l2], [peak_val_l2], s=150, color='#012696', edgecolor='white', linewidth=2, zorder=5)
plt.scatter([peak_date_l3], [peak_val_l3], s=150, color='#01847F', edgecolor='white', linewidth=2, zorder=5)

# Draw arrow from l2 peak to l3 peak
plt.annotate('',
             xy=(peak_date_l3, peak_val_l3),  # Arrow head (l3 peak)
             xytext=(peak_date_l2, peak_val_l2),  # Arrow tail (l2 peak)
             arrowprops=dict(arrowstyle='->,head_width=0.5,head_length=0.8',
                            color='#606060',
                            linewidth=2,
                            shrinkA=8,  # Space from start point
                            shrinkB=8,  # Space from end point
                            connectionstyle='arc3,rad=0'),  # Curved arrow
             zorder=4)

days_earlier = (peak_date_l2 - peak_date_l3).days
reduction = peak_val_l2 - peak_val_l3

annotation_text = (f'Peak {days_earlier} days earlier\n'
                  f'{reduction:.0f} fewer peak cases\n'
                  f'{cum_diff:.0f} fewer cumulative cases')

plt.text(peak_date_l3, peak_val_l3*0.75,  # Centered vertically
         annotation_text,
         ha='center', va='top',
         color='#333333',  # Unified dark color
         fontsize=14,
         linespacing=1.5,  # Space between lines
         bbox=dict(boxstyle='round,pad=0.3',
                  fc='white',
                  ec='none',  # No border
                  alpha=0))



# Existing economic improvement annotation
plt.annotate(
    'Economic\nImprovement',
    xy=(time[10], plt.ylim()[1]*0.2),
    xytext=(70,-15),
    textcoords='offset points',
    ha='center',
    va='bottom',
    color='#606060',
    fontsize=14,
    arrowprops=dict(
        arrowstyle='->,head_width=0.4,head_length=0.6',
        color='#606060',
        linewidth=2,
        shrinkA=0,
        shrinkB=0,
    )
)

plt.ylabel('Daily Cases', fontsize=16)
plt.xticks(time[::len(time) // 5], [t.strftime('%Y.%m') for t in time[::len(time) // 5]], fontsize=16)
plt.xlim([time[0], time[-1]])
plt.yticks(fontsize=16)

plt.legend([l1, l2, l3],
           ['Real Data', 'κ=0.2220', 'κ=0.3725'],
           loc='best', frameon=False, fontsize=14)
plt.title('Poland',
          fontsize=20, pad=20, weight='bold')
plt.tight_layout()
plt.savefig("../Fig/Contrast_Poland.jpg",
           dpi=600)
plt.show()

