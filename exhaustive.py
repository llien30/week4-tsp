from libs import read_csv, save_path, search_all_path

import argparse


def get_arguments():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(description="get number of csv file")

    parser.add_argument(
        "file_number", type=int, help="number of file to search shortest path"
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    n_cities, cities = read_csv(f"./google-step-tsp/input_{args.file_number}.csv")

    length, path = search_all_path(n_cities, cities)
    print(length)
    print(path)
    save_path(f"./google-step-tsp/output_{args.file_number}.csv", path)


if __name__ == "__main__":
    main()
