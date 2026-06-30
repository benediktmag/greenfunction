import matplotlib.pyplot as plt
from matplotlib import cm

def draw_plots(re, im, data, title="", cmap=cm.coolwarm):
    '''
    Plots the given data on a grid of size re*im, both as a contour and as a 3d graph
    '''
    fig2d, ax2d = plt.subplots()
    ax2d.set_aspect('equal', adjustable='box')
    CS = ax2d.contour(re, im, data, cmap=cmap)
    ax2d.clabel(CS, inline=1, fontsize=8)
    ax2d.set_title(f"{title} - contour plot")

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot_surface(re, im, data, rstride=1, cstride=1,
                      cmap=cmap, linewidth=0, antialiased=False)
    ax3d.set_title(f"{title} - graph")

    return fig2d, fig3d

def save_plots(plots, prefix):
    '''
    Saves generated plots as {prefix}_fig_{i}.pdf
    '''
    for i,plot in enumerate(plots):
        filename = f"{prefix}_fig_{i}.pdf"
        plot.savefig(filename, bbox_inches='tight')