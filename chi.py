from libs import (
    graham_scan,
    insert,
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
    n_cities, cities = read_csv(f"./google-step-tsp/input_{args.file_number}.csv")
    path = graham_scan(n_cities, cities)
    plot_path(path, cities, f"./image/graham_scan{args.file_number}.png")
    cost, path = insert(path, cities)
    print(f"CHI_cost : {cost}")
    cost1, swap_path = opt_2_swap(cities, n_cities, path)
    print(f"after 2-opt swap : {cost1}")

    cost2, reverse_path = opt_2_reverse(cities, n_cities, path)
    print(f"after 2-opt reverse : {cost2}")

    if cost1 < cost2:
        plot_path(swap_path, cities, f"./image/greedy_2opt_{args.file_number}.png")
        save_path(f"./google-step-tsp/output_{args.file_number}.csv", swap_path)

    elif cost2 <= cost1:
        plot_path(reverse_path, cities, f"./image/greedy_2opt_{args.file_number}.png")
        save_path(f"./google-step-tsp/output_{args.file_number}.csv", reverse_path)

    # print(path)


if __name__ == "__main__":
    main()
