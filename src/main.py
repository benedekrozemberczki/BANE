from bane import BANE
from parser import parameter_parser
from utils import read_graph, read_features, tab_printer


def main():
    args = parameter_parser()
    tab_printer(args)
    P = read_graph(args)
    X = read_features(args)
    model = BANE(args, P, X)
    model.fit()
    model.save_embedding()

if __name__ =="__main__":
    main()

