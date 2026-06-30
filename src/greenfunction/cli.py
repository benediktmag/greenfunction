import argparse
import json
import numpy as np

def safe_math_eval(expr_string, var_name="z"):
    """
    Safely evaluates strings from user to functions.
    Provides only abs, 'z' or 't' to evaluation context.
    """
    def math_func(var_val):
        allowed_context = {var_name: var_val, "abs": abs, "np": np}
        return eval(expr_string, {"__builtins__": {}}, allowed_context)
    return math_func

def parse_arguments():
    """Parses command line arguments and JSON config files."""

    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument('-f', '--file', type=str, help="Path to JSON configuration file.")

    args, _ = file_parser.parse_known_args()

    defaults = {}
    if args.file:
        try:
            with open(args.file, 'r') as f:
                defaults = json.load(f)

            if 'corner' in defaults and isinstance(defaults['corner'], str):
                defaults['corner'] = complex(defaults['corner'].replace(" ", ""))

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")

    parser = argparse.ArgumentParser(
        description="Calculate and visualize the nth approximation of a weighted Green function.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-f', '--file', type=str, help="Path to JSON configuration file.")

    parser.add_argument('-n', '--n_polys', type=int, default=50, help="Number of polynomials.")
    parser.add_argument('-c', '--corner', type=complex, default=-2-2j, help="Lower left corner point (e.g., -2-2j).")
    parser.add_argument('-w', '--width', type=float, default=4.0, help="Width of the graph.")
    parser.add_argument('-d', '--density', type=int, default=100, help="Density of the grid (N).")

    parser.add_argument('-b','--cond_expr', type=str, default="(abs(z.imag) <= 1) & (abs(z.real) <= 1)",
    help="Python expression string for the region condition.")
    parser.add_argument('-q','--q_expr', type=str, default="0 * z", help="Python expression string for the Q function.")
    parser.add_argument('--only_green', action='store_true', help="Only draw the Green's function")
    parser.add_argument('--hide', action='store_true', help="Do not display the generated plots")

    parser.add_argument('--use_boundary', action='store_true', help="Orthogonalize over a 1D boundary instead of a 2D region.")
    parser.add_argument('--curve_expr', type=str, default="0j + 0.9 * np.exp(1j * t)",
    help="Expression for boundary curve gamma(t), t in [0, 2pi)")
    parser.add_argument('--n_boundary', type=int, default=10000, help="Number of sample points on the boundary.")

    parser.add_argument('-s', '--save_settings', type=str, help="Save current settings to a JSON file (e.g., 'name' or 'name.json').")
    parser.add_argument('-o', '--output', type=str, default=None,
    help="Saves output with this prefix, an alias for --save_figs and --save_data.")
    parser.add_argument('--save_figs', type=str, nargs='?', const='green_approx', default=None,
    help="Save the generated figures as PDFs with this prefix.")
    parser.add_argument('--save_data', type=str, nargs='?', const='green_approx', default=None,
    help="Save the raw numpy computation results to an .npz file with this prefix.")
    parser.add_argument('-l', '--load_data', type=str, default=None,
    help="Path to an .npz file to load, skips computation.")

    parser.set_defaults(**defaults)

    return parser.parse_args()