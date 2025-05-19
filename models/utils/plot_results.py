import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def plot_confusion_matrix(confusion_mat, display_labels, fname=None):
    labels = list(map(lambda x: x[3:], display_labels))
    class_accuracy = confusion_mat.diagonal() / confusion_mat.sum(axis=1)
    pred_labels = [labels[i] + f" ({class_accuracy[i]:.2f})" for i in range(len(labels))]
        
    df = pd.DataFrame(confusion_mat, index=labels, columns=pred_labels)
    plt.figure(figsize=(7,5))
    ax = sn.heatmap(df.transpose(), cmap="YlGnBu", annot=True, fmt="d", cbar=False)
    plt.xlabel("Acutal class")
    plt.ylabel("Predicted class")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    # overall accuary
    # accuracy = np.trace(confusion_mat) / np.sum(confusion_mat)
    # print(accuracy)