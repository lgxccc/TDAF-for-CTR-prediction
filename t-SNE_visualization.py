import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Circle
import numpy as np
from matplotlib.patches import Ellipse
import pandas as pd

def t_sne(algorithm,x_for_tsne):
    algorithm.to('cpu')
    state = torch.load('D:\pyproject\EDG-DDA-Rec\domainbed\scripts/train_output\model_douban_erm.pkl')
    algorithm.load_state_dict(state["model_dict"])

    algorithm.eval()
    z_list = [algorithm.get_emb(x) for x in x_for_tsne]
    all_data = torch.cat(z_list, dim=0)
    labels = torch.cat([
        torch.zeros(z_list[0].size(0), dtype=torch.long),
        torch.ones(z_list[0].size(0), dtype=torch.long),
        torch.full((z_list[0].size(0),), 2, dtype=torch.long),
        torch.full((z_list[0].size(0),), 3, dtype=torch.long),
        torch.full((z_list[0].size(0),), 4, dtype=torch.long)
        # torch.full((z_list[0].size(0),), 5, dtype=torch.long)
    ])
    tsne = TSNE(n_components=2, random_state=0)
    reduced_data = tsne.fit_transform(all_data.detach().numpy())

    df = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
    df['Label'] = labels.numpy()

    colors = [(96/255,131/255,184/255), (0,155/255,131/255), (0,165/255,181/255), 'purple' , 'red']
    labels_dict = {0: 'T', 1: 'T+2', 2: 'T+4', 3: 'simulated', 4: 'ideal'}

    circle_params = []

    for i in range(5):
        if i%2==0 :
            subset = df[df['Label'] == i]
            center_x = subset['Dimension 1'].mean()
            center_y = subset['Dimension 2'].mean()
            radius = np.sqrt(((subset['Dimension 1'] - center_x) ** 2 + (subset['Dimension 2'] - center_y) ** 2).mean())
            circle_params.append((center_x, center_y, radius/10))

    x_6_center_x = circle_params[2][0] + circle_params[2][0] + abs(circle_params[1][0])
    x_6_center_y = 0.8 * circle_params[2][1] + 0.2 * abs(circle_params[1][1])
    x_6_center_radius = (circle_params[2][2] + circle_params[1][2] + circle_params[0][2])/3
    all_center_x = (circle_params[2][0] + circle_params[1][0] + circle_params[0][0] + x_6_center_x)/4
    all_center_y = (circle_params[2][1] + circle_params[1][1] + circle_params[0][1] + x_6_center_y)/4 -1.02
    all_center_radius = circle_params[2][2] *1.95

    # 绘制圆圈
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_facecolor('#F4F4F4')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)


    x_min, x_max = df['Dimension 1'].min(), df['Dimension 1'].max()
    y_min, y_max = df['Dimension 2'].min(), df['Dimension 2'].max()
    plt.xlim(x_min/7.5, x_max/7.5)
    plt.ylim(y_min/7.5, y_max/8)


    for (center_x, center_y, radius), color in zip(circle_params, colors):
        ellipse = Ellipse((center_x, center_y), width=2 * radius, height=2 * radius, color=color, fill=True, alpha=0.5)
        plt.gca().add_patch(ellipse)

    circle = Circle((x_6_center_x, x_6_center_y), x_6_center_radius, color='purple', fill=False, linestyle='--', linewidth=1)
    ellipse_all = Ellipse(xy=(all_center_x, all_center_y),width=all_center_radius*2,height=all_center_radius * 1.6,color='black',fill=False,linestyle='--',linewidth=1)
    plt.gca().add_patch(circle)
    plt.gca().add_patch(ellipse_all)

    font_properties = {'family': 'serif', 'size': 18}

    handles = [Ellipse((0, 0), width=2 * radius, height=2 * radius, color=color, fill=True, alpha=0.7) for radius, color
               in zip(circle_params, colors)]
    handles.append(Circle((0, 0), 0, color='purple', fill=False, linestyle='--', linewidth=1))
    handles.append(Ellipse((0, 0),width=all_center_radius*1.95, height=1.6 * all_center_radius, color='black', fill=False, linestyle='--', linewidth=1))
    plt.legend(handles=handles, labels=[labels_dict[i] for i in range(5)], ncol=5, fontsize=23, columnspacing=0.7, handletextpad=0.4, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig('general_.pdf', format='pdf')
    plt.show()
