import sys
from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from hipt_heatmap_utils import *
import os
import torch
from PIL import Image
import matplotlib
import argparse
import glob
import openslide
import numpy as np
import matplotlib.pyplot as plt

def generate_attention_maps(region, model, cmap, scale=1, alpha=0.5, threshold=None):

    x = eval_transforms()(region).unsqueeze(dim=0)

    print("generating attention maps...")
    hm4k, hm256, hm4k_256 = model.get_region_attention_heatmaps(x, fname="region", cmap = cmap, scale = scale, alpha=alpha, threshold=threshold)
    print("done")

    return hm4k, hm256, hm4k_256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',      type=str, default=None, help="path to input image")
    parser.add_argument('--wsi_path', type=str, default=None)
    parser.add_argument('--vit256',     type=str, default='HIPT_4K/Checkpoints/vit256_small_dino.pth')
    parser.add_argument('--vit4k',      type=str, default='HIPT_4K/Checkpoints/vit256_xs_dino.pth')
    parser.add_argument('--resolution', type=str, default="high", choices=["low", "high"])
    parser.add_argument('--threshold', type=float, default=None, help="threshold for attention map")
    parser.add_argument('--mode', type=str, default="single", choices=["single", "WSI"], help="whether to run attention map for just 1 crop or WSI")
    parser.add_argument('--save_dir',   type=str, default=None, help="where to save attention maps")

    args = parser.parse_args()

    light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)

    ### ViT_256 + ViT_4K loaded into HIPT_4K API
    model = HIPT_4K(model256_path=args.vit256, model4k_path=args.vit4k)
    model.eval()

    if args.mode == "single":
        scale = 1
    else:
        scale = 16


    if args.mode == "single":

        region = Image.open(args.image).convert('RGB')

        slide_name = args.image.split("/")[-2]

        hm4k, hm256, hm4k_256 = generate_attention_maps(region, model, light_jet, scale=scale)

        region = region.convert('RGBA')

        print("saving attention maps...")

        os.makedirs(os.path.join(args.save_dir, slide_name), exist_ok=True)

        for head_idx in range(len(hm4k_256)):

            att_map = np.array(hm4k_256[head_idx])
            #print(att_map.shape)

            plt.imsave(os.path.join(args.save_dir, slide_name, f"{slide_name}_head_{head_idx}.png"), att_map, cmap='jet')

            att_map = Image.open(os.path.join(args.save_dir, slide_name, f"{slide_name}_head_{head_idx}.png"))

            att_map = att_map.convert('RGBA')

            fused = Image.blend(region,att_map, 0.3)
            fused.save(os.path.join(args.save_dir, slide_name, f"{slide_name}_head_{head_idx}.png"), "PNG")

        print("done")

    else:

        # collect all 4k heatmaps and stitch them
        # read in all crops
        form = ".tiff"
        level=0

        region_paths = glob.glob(os.path.join(args.image, "*.png"))

        slide_name = args.image.split("/")[-1]

        # read in the WSI
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, slide_name+form))
        # set the level
        level = level

        # get the dimensions of the WSI at level 0
        dimensions = slide.dimensions

        # define the size of patches
        patch_size = 4096
        png_step = 4096 // scale

        # create a large png of the slide
        png = slide.get_thumbnail((dimensions[0] // scale, dimensions[1] // scale))

        png_rows = dimensions[1] // scale
        png_cols = dimensions[0] // scale

        # read in the tissue map, crete binary of it and multiply with the attention map at the end
        qc = Image.open( os.path.join(qc_path, slide_name+".tif") )
        qc = qc.resize((png_cols, png_rows))

        image_file = qc.convert('L')
        image_file = image_file.point( lambda p: 0 if p == 179 else 255 )
        # To mono
        image_file = image_file.convert('1')

        qc = np.array(image_file)

        for head_idx in range(6):

            att_map = np.zeros((png_rows, png_cols))

            for region_path in region_paths:
                region = Image.open(region_path).convert('RGB')

                hm4k, hm256, hm4k_256 = generate_attention_maps(region, model, matplotlib.cm.jet, scale=scale, alpha=1, threshold=args.threshold)
                
                if args.resolution == "high":
                    heatmap = np.array(hm4k_256[(head_idx*6) + 3])
                else:
                    heatmap = np.array(hm4k[head_idx])


                print(region_path.split("/")[-1].split(".")[0].split("_")[-1].split("-")[0])

                region_row, region_col = int(region_path.split("/")[-1].split(".")[0].split("_")[-1].split("-")[1]) // scale, int(region_path.split("/")[-1].split(".")[0].split("_")[-1].split("-")[0]) // scale

                att_map[region_row:region_row+png_step, region_col:region_col+png_step] = heatmap

            os.makedirs(os.path.join(args.save_dir, slide_name), exist_ok=True)

            att_map = att_map * qc

            att_map = Image.fromarray(att_map.astype(np.uint8))

            plt.imsave(os.path.join(args.save_dir, slide_name, f"{slide_name}_head_{head_idx}.png"), att_map, cmap='jet')

            att_map = Image.open(os.path.join(args.save_dir, slide_name, f"{slide_name}_head_{head_idx}.png"))

            png = png.convert('RGBA')
            att_map = att_map.convert('RGBA')

            fused = Image.blend(png,att_map, 0.3)
            fused.save(os.path.join(args.save_dir, slide_name, f"{slide_name}_head_{head_idx}.png"), "PNG")




if __name__ == '__main__':
    main()