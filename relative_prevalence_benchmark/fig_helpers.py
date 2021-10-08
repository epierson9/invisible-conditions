from matplotlib.patches import PathPatch
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
