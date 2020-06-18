from libs import (
    greedy_search,
    opt_2_reverse,
    opt_2_swap,
    read_csv,
    save_path,
    plot_path,
)

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
    n_cities, cities = read_csv(f"input_{args.file_number}.csv")

    cost, path = greedy_search(n_cities, cities)
    print(f"Greedy cost: {cost}")
    plot_path(path, cities, f"./image/greedy_{args.file_number}.png")

    cost1, swap_path = opt_2_swap(cities, n_cities, path)
    print(f"After 2-opt swap: {cost1}")
    plot_path(swap_path, cities, f"./image/greedy_2opt-swap_{args.file_number}.png")

    cost2, reverse_path = opt_2_reverse(cities, n_cities, path)
    print(f"After 2-opt reverse: {cost2}")
    plot_path(
        reverse_path, cities, f"./image/greedy_2opt-reverse_{args.file_number}.png"
    )

    if cost1 < cost2:
        save_path(f"solution_yours_{args.file_number}.csv", swap_path)

    else:
        save_path(f"solution_yours_{args.file_number}.csv", reverse_path)


if __name__ == "__main__":
    main()
