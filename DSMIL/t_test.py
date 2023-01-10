import numpy as np
from scipy.stats import ttest_ind
import argparse

def parse(path):
    with open(path) as f:
        contents = f.readlines()

    fold_results = contents[1:6]

    a = np.zeros(5)

    for i in range(len(fold_results)):
        auc_value = fold_results[i].split(":")[2].split(" ")[1]
        a[i] = auc_value

    return a
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs=2, default=None, help="path to 2 results files")

    args = parser.parse_args()

    # parse the results file
    a = parse(args.data[0])
    b = parse(args.data[1])

    _, p = ttest_ind(a,b,equal_var=False)

    print("For the data: ", args.data)
    print("The p value is: ", p)



if __name__ == '__main__':
    main()