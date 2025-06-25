import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap


countries_data = [
    {'name': 'South Africa','HDI': 0.699, 'Classification': 0},
    {'name': 'Namibia','HDI': 0.647, 'Classification': 0},
    {'name': 'Morocco','HDI': 0.667, 'Classification': 0},
    {'name': 'Zambia', 'HDI': 0.588, 'Classification': 0},
    {'name': 'Mozambique', 'HDI': 0.437, 'Classification': 0},
    {'name': 'Singapore', 'HDI': 0.932, 'Classification': 1},
    {'name': 'Japan','HDI': 0.919, 'Classification': 1},
    {'name': 'India','HDI': 0.640, 'Classification': 0},
    {'name': 'Myanmar','HDI': 0.578, 'Classification': 0},
    {'name': 'Germany','HDI': 0.942, 'Classification': 1},
    {'name': 'United Kingdom','HDI': 0.922, 'Classification': 1},
    {'name': 'France','HDI': 0.901, 'Classification': 1},
    {'name': 'Spain','HDI': 0.876, 'Classification': 1},
    {'name': 'Hungary','HDI': 0.828, 'Classification': 1},
    {'name': 'Poland','HDI': 0.843, 'Classification': 1},
    {'name': 'Romania','HDI': 0.811, 'Classification': 1},
    {'name': 'Chile','HDI': 0.832, 'Classification': 1},
    {'name': 'Argentina','HDI': 0.825, 'Classification': 1},
    {'name': 'Brazil','HDI': 0.754, 'Classification': 0},
    {'name': 'Bolivia','HDI': 0.693, 'Classification': 0},
    {'name': 'Australia','HDI': 0.939, 'Classification': 1},
    {'name': 'Sweden','HDI': 0.907, 'Classification': 1},
    {'name': 'Italy','HDI': 0.873, 'Classification': 1},
    {'name': 'Lithuania','HDI': 0.839, 'Classification': 1},
    {'name': 'Ukraine','HDI': 0.747, 'Classification': 0},
    {'name': 'Croatia','HDI': 0.831, 'Classification': 1},
    {'name': 'Russia','HDI': 0.816, 'Classification': 1},
    {'name': 'Saudi Arabia','HDI': 0.853, 'Classification': 1},
    {'name': 'Turkey','HDI': 0.820, 'Classification': 1},
    {'name': 'Iraq','HDI': 0.685, 'Classification': 0},
    {'name': 'Israel','HDI': 0.903, 'Classification': 1},
    {'name': 'Nepal','HDI': 0.574, 'Classification': 0},
    {'name': 'Kazakhstan','HDI': 0.800, 'Classification': 0},
    {'name': 'Philippines','HDI': 0.699, 'Classification': 0},
    {'name': 'Bangladesh','HDI': 0.632, 'Classification': 0},
    {'name': 'Pakistan','HDI': 0.562, 'Classification': 0},
    {'name': 'Thailand','HDI': 0.755, 'Classification': 0},
    {'name': 'Korea','HDI': 0.903, 'Classification': 1},
    {'name': 'Mexico','HDI': 0.756, 'Classification': 0},
    {'name': 'Colombia','HDI': 0.747, 'Classification': 0},
    {'name': 'Costa Rica','HDI': 0.794, 'Classification': 0},
    {'name': 'Panama','HDI': 0.780, 'Classification': 0},
    {'name': 'Peru','HDI': 0.750, 'Classification': 0},
    {'name': 'Austria','HDI': 0.908, 'Classification': 1},
    {'name': 'Bulgaria','HDI': 0.782, 'Classification': 0},
    {'name': 'Armenia','HDI': 0.743, 'Classification': 0},
    {'name': 'Azerbaijan','HDI': 0.751, 'Classification': 0},
    {'name': 'Czech Republic','HDI': 0.888, 'Classification': 1},
    {'name': 'Iran','HDI': 0.798, 'Classification': 0},
    {'name': 'Ireland','HDI': 0.942, 'Classification': 1},
    {'name': 'Jordan','HDI': 0.735, 'Classification': 0},
    {'name': 'Netherlands','HDI': 0.931, 'Classification': 1},
    {'name': 'Portugal','HDI': 0.830, 'Classification': 1},
    {'name': 'Qatar', 'HDI': 0.856, 'Classification': 1},
    {'name': 'Serbia', 'HDI': 0.771, 'Classification': 0},
    {'name': 'Switzerland', 'HDI': 0.930, 'Classification': 1},
    {'name': 'Canada', 'HDI': 0.926, 'Classification': 1},
    {'name': 'America', 'HDI': 0.920, 'Classification': 1},
]


sorted_data = sorted(countries_data, key=lambda x: x['HDI'], reverse=True)
sorted_countries = [x['name'] for x in sorted_data]
sorted_hdi = [x['HDI'] for x in sorted_data]
classifications = [x['Classification'] for x in sorted_data]




sns.set_style("white")
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.labelsize': 11,
})


fig, ax = plt.subplots(figsize=(8, 10))
ax.set_ylim(-1.2, 58 + 0.2)


color_1 = '#073E7F'
color_2 = '#9B9C2D'
colors = [color_1 if c == 1 else color_2 for c in classifications]


bars = ax.barh(sorted_countries, sorted_hdi, color=colors, height=0.7)
ax.invert_yaxis()

gradient = LinearSegmentedColormap.from_list('my_gradient', ['#FFFFFF', '#9B9C2D'])

kaz_index = sorted_countries.index('Kazakhstan')
divider_pos = kaz_index - 0.5
ax.axhline(y=divider_pos, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
ax.text(0.99, divider_pos + 1.2, ' HDI threshold (0.80~0.81)',
        ha='right', va='bottom', color='gray', fontstyle='italic', fontsize=10)


for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            ha='left', va='center', fontsize=9)


legend_elements = [
    Patch(facecolor=color_1, label='Benefit-driven (linear)'),
    Patch(facecolor=color_2, label='Barrier-inhibition (nonlinear)')
]
ax.legend(handles=legend_elements,
          loc='lower right',
          # title='Classification',
          frameon=False,
          title_fontsize=10,
          fontsize=12,
          )


ax.set_xlim(0.4, 1)



plt.tight_layout()


plt.savefig(
    '../Fig/HDI.tiff',
    dpi=1800,
    format='tiff',
    bbox_inches='tight',
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.show()