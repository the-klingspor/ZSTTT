import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def embed_2D(features, n_neighbors=100):
    # initialize a UMAP model
    umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors)

    # fit the UMAP model to the feature vectors and obtain the 2D embedding
    embedding = umap_model.fit_transform(features)

    return embedding


def plot_embedding(features, labels, seen_classes, unseen_classes, save_dir=None, logger=None):
    # get the number of samples
    n_samples = len(labels)

    # get the permutation index
    permutation_idx = np.random.permutation(n_samples)

    # shuffle the labels and features using the same permutation index
    shuffled_labels = labels[permutation_idx]
    shuffled_features = features[permutation_idx]

    # create a list of all unique labels
    all_labels = np.unique(labels)

    # create a custom color palette with shades of red and blue for the seen and unseen classes
    seen_palette = sns.color_palette("Reds", n_colors=len(seen_classes), desat=0.8)
    unseen_palette = sns.color_palette("Blues", n_colors=len(unseen_classes), desat=0.8)
    label_palette = {}
    for i, label in enumerate(all_labels):
        if label in seen_classes:
            label_palette[label] = seen_palette[list(seen_classes).index(label)]
        elif label in unseen_classes:
            label_palette[label] = unseen_palette[list(unseen_classes).index(label)]

    # create a scatter plot of the embedded features, using the custom color palette
    sns.scatterplot(x=shuffled_features[:, 0], y=shuffled_features[:, 1], hue=shuffled_labels, palette=label_palette)

    # create custom legend with representative for seen and unseen classes
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='seen', markerfacecolor='r', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='unseen', markerfacecolor='b', markersize=10)]

    # add the legend to the plot
    plt.legend(handles=legend_elements)

    if logger:
        plt_dict = {"embeddings": plt}
        logger.log(plt_dict)

    # save the plot to a file if a save directory is specified
    if save_dir:
        plt.savefig(save_dir)

    plt.show()

def embed_and_plot(data, save_dir=None, logger=None):
    embeddings = embed_2D(data.feature)

    plot_embedding(embeddings, data.label, data.seenclasses, data.unseenclasses, save_dir=save_dir, logger=logger)
