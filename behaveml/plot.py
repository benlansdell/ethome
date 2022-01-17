import matplotlib.pyplot as plt

def plot_tsne(tsne_embedding, color_data, size = 1, figsize = (8,8), legend_title = 'prediction'):

    fig, axes = plt.subplots(1,1, figsize = figsize)
    sctr = axes.scatter(x = tsne_embedding[:,0], y = tsne_embedding[:,1], c = color_data, s = size)
    axes.set_xlabel('TSNE dim 1')
    axes.set_ylabel('TSNE dim 2')
    legend1 = axes.legend(*sctr.legend_elements(), loc="lower left", title=legend_title)
    axes.add_artist(legend1)
    return fig, axes