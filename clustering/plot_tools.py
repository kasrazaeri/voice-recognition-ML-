import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.cm as cm


def plot_3d(data: np.array, label = None, title = None, s:int = 2):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(projection='3d')

    if not title is None:
        plt.title(title)

    if not label is None:
        n = len(set(label))
        x = np.arange(n)
        ys = [i+x+(i*x)**2 for i in range(n)]
        colors = cm.rainbow(np.linspace(0, 1, len(ys)))

        print(f"[INFO] plotting {n} classes.")
        for i, l in enumerate(set(label)):
            g = data[label == l, :]

            ax.scatter(g[:, 0], g[:, 1], g[:, 2], s=s, color=colors[i])
    else:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=s)

