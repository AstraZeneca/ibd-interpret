"""

Parse all the 256x256 patches out of a 4096x4096 patch and save

"""

print("loading packages")
from PIL import Image
import numpy as np
import glob
import argparse
import random
import os
import time
import pandas as pd
import time
print("done")

def parse_4k(region_path):

    # load in region and convert to np array
    region = np.array(Image.open(region_path))

    # define a tensor to hold the patches
    patches_256 = np.zeros((256,256,256,3))

    # counter counts from 0 to 255 for indexing
    counter = 0
    for i in range(16):
        # for each row (16 rows)
        # row index multiplied by 256 to get patch coordinate
        ix = i * 256
        for j in range(16):
            # for each column (16 columns)
            # column index multiplied by 256 to get patch coordinate
            jx = j * 256
            # get patch
            patch = region[ix:ix+256, jx:jx+256, :]
            patches_256[counter] = patch
            counter+=1

    return patches_256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", type=str, default=None, help="path to 4k patches")
    parser.add_argument("--save_dir", type=str, default=None, help="where to save 256x256 patches")

    args = parser.parse_args()

    bag_list = os.listdir(args.patch_dir)

    random.shuffle(bag_list)

    if "extracted_mag20x_patch4096_fp/vits_tcga_pancancer_dino_pt_patch_features_384" in bag_list:
        bag_list.remove("extracted_mag20x_patch4096_fp/vits_tcga_pancancer_dino_pt_patch_features_384")


    for i in range(len(bag_list)):

        print(f"parsing patches for: {bag_list[i]}")        
        region_paths = glob.glob(os.path.join(args.patch_dir, bag_list[i], "*.png"))

        print(f"number of 4k regions to parse: {len(region_paths)}")

        start = time.time()

        for region_path in region_paths:

            print("parsing patches for ", region_path)

            try:

                patches_256 = parse_4k(region_path)

            except Exception as e:
                print(e)
                print("could not parse patches for ", region_path)
                continue

            print("patches extracted, saving...")

            for j in range(256):
                patch_256 = Image.fromarray(patches_256[j,:,:,:].astype(np.uint8)).convert('RGB')
                patch_256.save( os.path.join(args.save_dir, region_path.split("/")[-1].split(".")[0] + f"_patch_{str(j)}.png" ) )

            print("done")

        end = time.time()

        print("done")
        print(f"completed parsing for {i+1} / {len(bag_list)}, progress: { ((i+1)/len(bag_list)) * 100}, took {end-start} seconds")

if __name__ == '__main__':
    main()