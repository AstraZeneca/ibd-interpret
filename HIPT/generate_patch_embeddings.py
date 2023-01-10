from PIL import Image
from HIPT_4K.hipt_4k import HIPT_4K
from HIPT_4K.hipt_model_utils import eval_transforms
import torch
import sys
import numpy as np
import argparse
import os
import glob
import random


# function for generating embedding for a 4k patch
def generate_embedding(region, model):

    x = eval_transforms()(region).unsqueeze(dim=0)
    out = model.generate_vit256_embeddings(x)

    embedding = out.cpu().numpy()

    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, default=None, help="path to where patches are")
    parser.add_argument('--feats_dir', type=str, default=None, help='Path to where embeddings are stored')
    parser.add_argument('--checkpoint', type=str, default='HIPT_4K/Checkpoints/vit256_small_dino.pth', help="checkpoint to use, defaults to authors")
    args = parser.parse_args()

    # get the list of bags
    bag_list = glob.glob( os.path.join(args.patch_dir, '*') )

    print("total WSIs to be processed:", len(bag_list))

    # model init
    model = HIPT_4K(model256_path=args.checkpoint, device4k='cuda:0')
    model.eval()

    random.shuffle(bag_list)

    if not os.path.exists(args.feats_dir):
        os.makedirs(args.feats_dir, exist_ok=True)


    for bag in bag_list:
        
        # since the embeddings folder is in the same folder as the patches, 
        # we should ignore the embeddings folder
        if "extracted" in bag:
            continue

        # get the list of patches for this bag
        regions = glob.glob( os.path.join(bag, "*") )

        print("processing", bag, "num regions:", len(regions))

        for i in range(len(regions)):

            try:
                # try to open patch
                region = Image.open(regions[i]).convert('RGB')
            except Exception as e:
                # log
                print("region could not be loaded:", regions[i])
                print(e)
                continue

        for region_path in regions:

            if not os.path.exists( os.path.join(args.feats_dir, region_path.split("/")[-1].split(".")[0] + ".pt") ):


                try:
                    # try to open patch
                    region = Image.open(region_path).convert('RGB')
                except Exception as e:
                    # log
                    print("region could not be loaded:", regions[i])
                    print(e)
                    continue

                patch_256_embeddings = generate_embedding(region, model)

                # save embeddings as a pt file
                torch.save(torch.from_numpy(patch_256_embeddings), os.path.join(args.feats_dir, region_path.split("/")[-1].split(".")[0] + ".pt"))

            else:
                print("skipping, features already generated")
                continue


if __name__ == '__main__':
    main()