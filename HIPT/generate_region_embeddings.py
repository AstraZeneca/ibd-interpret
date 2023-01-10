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
    """generate_embedding

    Args:
        region (PIL.Image): image to run inference on
        model (HIPT_4K): HIPT model

    Returns:
        numpy.ndarray: embedding vector for the image
    """

    x = eval_transforms()(region).unsqueeze(dim=0)
    out = model.forward(x)

    embedding = out.cpu().numpy()

    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir',  type=str, default=None, help="path to where patches are")
    parser.add_argument("--vit256",     type=str, default='HIPT_4K/Checkpoints/vit256_small_dino.pth', help="path to checkpoint")
    parser.add_argument("--vit4k",      type=str, default='HIPT_4K/Checkpoints/vit4k_xs_dino.pth', help="path to checkpoint")
    parser.add_argument('--feats_dir',  type=str, default=None, help='Path to where embeddings are stored')
    args = parser.parse_args()

    # get the list of bags
    bag_list = glob.glob( os.path.join(args.patch_dir, '*') )

    print("total WSIs to be processed:", len(bag_list))

    # model init
    model = HIPT_4K(model256_path=args.vit256, model4k_path=args.vit4k)
    model.eval()

    random.shuffle(bag_list)

    if not os.path.exists(args.feats_dir):
        os.makedirs(args.feats_dir, exist_ok=True)


    for bag in bag_list:
        
        # since the embeddings folder is in the same folder as the patches, 
        # we should ignore the embeddings folder
        if bag == args.feats_dir:
            continue

        if not os.path.exists( os.path.join(args.feats_dir, bag.split("/")[-1] + ".pt") ):

            # get the list of patches for this bag
            regions = glob.glob( os.path.join(bag, "*") )
            # init a matrix that holds embeddings for all patches
            patch_array = np.zeros((len(regions), 192))

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


                # generate the embedding
                embedding = generate_embedding(region, model)

                # store
                patch_array[i] = embedding


            # save embeddings as a pt file
            torch.save(torch.from_numpy(patch_array), os.path.join(args.feats_dir, bag.split("/")[-1]+".pt"))

        else:
            print("skipping, features already generated")
            continue


if __name__ == '__main__':
    main()