import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help="path to raw splits directory")
    parser.add_argument('--save_path', type=str, default=None, help="where to save HIPT splits")

    args = parser.parse_args()

    # for split in splits
    # create a split csv
    # copy train and test into split csv
    
    num_splits = len(os.listdir(args.data))

    for i in range(num_splits):

        current_train = pd.read_csv(os.path.join(args.data, str(i), f"X_train_fold_{i}.csv"))
        current_test = pd.read_csv(os.path.join(args.data, str(i), f"X_test_fold_{i}.csv"))

        output = pd.DataFrame()
        output["train"] = pd.Series(current_train.iloc[:,0].to_list())
        output["test"] = pd.Series(current_test.iloc[:,0].to_list())
        output["val"] = pd.Series(dtype='float64')

        output.to_csv(os.path.join(args.save_path, f"splits_{i}.csv"), index=False)



if __name__ == '__main__':
    main()