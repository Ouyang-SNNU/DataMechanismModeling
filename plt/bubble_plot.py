import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from scipy import stats


# Read the Excel file
df = pd.read_excel('../Data/policy adjustment.xlsx', sheet_name='linear')

# Extract each column as a separate variable
country = np.array(df['Countries'])
kappa = np.array(df['kappa'])
m = np.array(df['m'])
population = np.array(df['Population'])
gdp_per_capita = np.array(df['GDP per capita'])
hdi = np.array(df['human_development_index'])


inv_m = 1 / m
a= kappa * m / population


plt.figure(figsize=(10.2, 8.2))
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

slope, intercept, r_value, p_value, std_err = stats.linregress(gdp_per_capita, a)
x_fit = np.linspace(gdp_per_capita.min(), gdp_per_capita.max(), 100)
y_fit = intercept + slope * x_fit

sns.regplot(
    x=gdp_per_capita,
    y=a,
    scatter=False,
    ci=95,
    line_kws={
        'color': 'red',
        'lw': 1.5,
        'ls': '--',
        'label': f'Linear fit (p={stats.pearsonr(gdp_per_capita, a)[1]:.3f})'
    },
    scatter_kws={'alpha': 0},
)


scatter = plt.scatter(
    x=gdp_per_capita,
    y=a,
    s=(kappa / kappa.max() * 800) + 100,
    c=m,
    cmap='coolwarm',
    alpha=0.7,
    edgecolors='w',
    linewidth=0.5
)


plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


texts = []
for i, txt in enumerate(country):
    texts.append(plt.text(
        x=gdp_per_capita[i],
        y=a[i],
        s=txt,
        fontsize=10,
        alpha=0.8,
        ha='center',
        fontweight='bold',
        va='center'
    ))



adjust_text(
    texts,
    arrowprops=dict(arrowstyle='-', color='gray', lw=1, linestyle=':'),
    expand_points=(1.5, 1.5),
    expand_text=(1.2, 1.2),
    force_text=0.7,
    only_move={'points':'y', 'text':'xy'}
)



kappa_max = kappa.max()


sizes_legend = [0.25, 0.5, 0.75, 1.0]
bubble_area = [(s * 800 + 100) for s in sizes_legend]


fig, ax = plt.gcf(), plt.gca()
ax.tick_params(axis='x', labelsize=26)
ax.tick_params(axis='y', labelsize=26)
ax.yaxis.get_offset_text().set_fontsize(22)


x0, y0 = 0.05, 0.58
gap = 0.1


for i, (s, area) in enumerate(zip(sizes_legend, bubble_area)):
    radius_pts = np.sqrt(area)
    radius_ax = (radius_pts / 72) / fig.get_size_inches()[1]

    y = y0 + i * gap + radius_ax


    ax.plot(
        x0, y,
        marker='o',
        markersize=radius_pts,
        markerfacecolor='none',
        markeredgecolor='gray',
        markeredgewidth=1.0,
        alpha=0.9,
        transform=ax.transAxes,
        clip_on=False
    )


    label = (
        fr'{s:.2f} Max $\kappa$' if s < 1.0 else r'Max $\kappa$'
    )
    ax.text(
        x0 + 0.035, y,
        label,
        transform=ax.transAxes,
        va='center',
        fontsize=16
    )



plt.tight_layout()
plt.savefig(
    '../Fig/bubble_plot.tif',
    dpi=600,
    format='tiff',
    bbox_inches='tight',
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.show()