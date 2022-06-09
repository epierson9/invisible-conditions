from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches
import numpy as np

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

def add_metrics_to_df(results_df):
    results_df['rel_prior_err'] = results_df['true_rel_prior'] - results_df['pred_rel_prior']
    results_df['g1_prior_err'] = results_df['true_g1_prior'] - results_df['pred_g1_prior']
    results_df['g2_prior_err'] = results_df['true_g2_prior'] - results_df['pred_g2_prior']

    results_df['rel_prior_err_pct'] = results_df['pred_rel_prior']/results_df['true_rel_prior']
    results_df['log_rel_prior_err_pct'] = np.log2(results_df['rel_prior_err_pct'])
    results_df['g1_prior_err_pct'] = results_df['pred_g1_prior']/results_df['true_g1_prior']
    results_df['log_g1_prior_err_pct'] = np.log2(results_df['g1_prior_err_pct'])
    results_df['g2_prior_err_pct'] = results_df['pred_g2_prior']/results_df['true_g2_prior']
    results_df['log_g2_prior_err_pct'] = np.log2(results_df['g2_prior_err_pct'])
    return results_df

def change_box_colors(ax, method_colors):
    # Change the color of the entire box to be method color
    for j,box in enumerate(ax.artists):
        color = method_colors[j%len(method_colors)]
        box.set_edgecolor(color)
        box.set_facecolor('white')
        box.set_linecolor(color)
        for k in range(6*j,6*(j+1)):
             ax.lines[k].set_color(color)

def change_legend_labels(g, methods, method_name_dict):
    new_labels = [method_name_dict[method_name] for method_name in methods]
    for t, l in zip(g.get_legend().texts, new_labels): t.set_text(l)
        

def make_legend(ax, methods, method_colors, method_name_dict, loc='lower right'):
    patches = []
    for method_name, method_color in zip(methods, method_colors):
        patches.append(mpatches.Patch(color=method_color, label=method_name))
    labels = [method_name_dict[m] for m in methods]
    ax.legend(handles = patches, labels = labels,loc=loc)
