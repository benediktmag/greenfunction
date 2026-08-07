import matplotlib.pyplot as plt
from greenfunction.cli import parse_arguments, safe_math_eval
from greenfunction.computing import grid_points, region_points, boundary_points, green_approx, george_approx, dirichlet_szego
from greenfunction.plotting import draw_plots, save_plots
from greenfunction.io import save_settings, save_data, load_data


def main():
    '''
    Computes a numerical approximation to the Green's function according to user specified settins and plots them
    '''
    args=parse_arguments()
    if args.save_settings:
        save_settings(args.save_settings, vars(args))

    if args.load_data:
        loaded_data = load_data(args.load_data)
        re, im = loaded_data['Re'], loaded_data['Im']
        green = loaded_data['green']
        george = loaded_data['george'] if 'george' in loaded_data else None
        dirichlet = loaded_data['dirichlet'] if 'dirichlet' in loaded_data else None
    else:
        condition = safe_math_eval(args.cond_expr)
        q_func = safe_math_eval(args.q_expr)

        re, im, z_grid = grid_points(args.corner, args.width, args.density)

        dirichlet = None

        if args.use_boundary:
            gamma_func = safe_math_eval(args.curve_expr, var_name="t")
            region = boundary_points(gamma_func, args.n_boundary)

            if args.boundary_vals:
                boundary_func= safe_math_eval(args.boundary_vals, var_name="z")
                dirichlet = dirichlet_szego(grid=z_grid, gamma=gamma_func, boundary_func=boundary_func, n=args.density,
                boundary_n=args.n_boundary, q_func=q_func)

        else:
            region = region_points(
                z_grid, args.width, condition, args.density
            )

        green = green_approx(z_grid, args.n_polys, region, q_func)
        george = None
        if not args.only_green:
            george = george_approx(z_grid, args.n_polys, region, q_func)

        if args.save_data:
            data_dict = {'Re': re, 'Im': im, 'green': green}
            if george is not None:
                data_dict['george'] = george
            if dirichlet is not None:
                data_dict['dirichlet'] = dirichlet
            save_data(args.save_data, **data_dict)

    figs = []

    fig1, fig2 = draw_plots(re, im, green, "Green's function")
    figs.extend([fig1, fig2])

    if george is not None:
        fig3, fig4 = draw_plots(re, im, george, "George's function")
        figs.extend([fig3, fig4])

    if dirichlet is not None:
        fig5, fig6 = draw_plots(re, im, dirichlet, "Solution to the given Dirichlet problem")
        figs.extend([fig5, fig6])

    if args.save_figs:
        save_plots(figs, args.save_figs)

    if not args.hide:
        plt.show()

if __name__ == "__main__":
    main()