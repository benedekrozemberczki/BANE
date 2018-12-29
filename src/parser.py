import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the Twitch Brasilians dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """

    parser = argparse.ArgumentParser(description = "Run BANE.")


    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/ptbr_edges.csv",
	                help = "Edge list csv.")

    parser.add_argument("--feature-path",
                        nargs = "?",
                        default = "./input/ptbr_features.json",
	                help = "Node features csv.")

    parser.add_argument("--output-path",
                        nargs = "?",
                        default = "./output/ptbr_bane.csv",
	                help = "Target embedding csv.")

    parser.add_argument("--features",
                        nargs = "?",
                        default = "sparse",
	                help = "Feature matrix structure.")

    parser.add_argument("--dimensions",
                        type = int,
                        default = 48,
	                help = "Number of SVD factors. Default is 48.")

    parser.add_argument("--binarization-rounds",
                        type = int,
                        default = 10,
	                help = "Number of power iterations. Default is 10.")

    parser.add_argument("--approximation-rounds",
                        type = int,
                        default = 5,
	                help = "Number of CDC rounds. Default is 5.")

    parser.add_argument("--order",
                        type = int,
                        default = 1,
	                help = "Adjacency matrix power in target creation. Default is 1.")

    parser.add_argument("--gamma",
                        type = float,
                        default = 0.7,
	                help = "Trade-off parameter. Default is 0.7.")

    parser.add_argument("--alpha",
                        type = float,
                        default = 0.01,
	                help = "Regularization parameter. Default is 0.01.")
    
    return parser.parse_args()
