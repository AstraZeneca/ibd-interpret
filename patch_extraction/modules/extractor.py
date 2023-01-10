import os
from patch_extractor import PatchExtractor
import argparse
import random
import yaml

def extract_patches(args, config, method="grid_extract"):
    """
    extract_patches - driver function for extracting patches

    Args:
        args (_type_): _description_
        config (_type_): yaml config file
        method (str, optional): which patch extraction method to use. Defaults to "grid_extract".
    """

    WSI_filenames = os.listdir(args.wsi_path)

    if '.DS_Store' in WSI_filenames:
        WSI_filenames.remove('.DS_Store')

    WSI_filenames = sorted(WSI_filenames)

    random.shuffle(WSI_filenames)

    # print(WSI_filenames)
    for name in WSI_filenames:
        print(name)

    # optional - specify a list of WSIs to ignore
    ignore_list = []
        
    for filename in WSI_filenames:
        
        print(filename)
        
        # if ignored slide encountered, skip
        if filename in ignore_list:
            continue
        
        # if a folder for this slide does not already exist and the file doesn't end in .partial, create patches
        if not os.path.exists( os.path.join(args.processed_path, filename.split(".")[0] ) ) and filename.split(".")[-1] != "partial":

            # create an extractor
            extractor = PatchExtractor(
                wsi_path=os.path.join(args.wsi_path, filename), 
                processed_path=args.processed_path, 
                patch_size=(config['patch_size'], config['patch_size']), 
                level=config['level'], 
                seed=42,
                alpha=config['alpha'],
                check_tissue=config['check_tissue'], 
                tissue_percent=config['tissue_percent'],
                prefix=filename.split(".")[0] + "/", 
                suffix=config['suffix']
            )

            if method == "grid_extract":

                # perform extraction
                extractor.grid_extract(
                    pixel_overlap=round(config['pixel_overlap']* config['patch_size']), 
                    get_mask=config['get_mask'], 
                    get_patch_locations=config['get_patch_locations'], 
                    save_meta=args.meta_path, 
                    save_patches=config['save_patches']
                )

            elif method == "HALO_grid_extract":
                extractor.HALO_grid_extract(
                    args.QC_path, 
                    pixel_overlap=round(config['pixel_overlap']* config['patch_size']), 
                    get_mask=config['get_mask'], 
                    get_patch_locations=config['get_patch_locations'], 
                    save_meta=args.meta_path, 
                    save_patches=config['save_patches']
                )

            else:
                raise("Error - valid extraction method not specified, choices are grid_extract and HALO_grid_extract ")

        else:
            print("already being tiled, skipping...")
            continue
    
    # count how many patches were produced
    total_patches = 0
    for _, _, files in os.walk(args.processed_path):   
        total_patches += len(files)

    print(f"total patches extracted from {len(WSI_filenames)} WSIs: {total_patches}")


def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, help="path to where to save meta data")
    parser.add_argument("--wsi_path", type=str, help="path to WSIs")
    parser.add_argument("--processed_path", type=str, help="path to where to save patches")
    parser.add_argument("--config_path", type=str, help="path to extractor config file")
    parser.add_argument("--QC_path", type=str, default="", help="path to QC results")

    args = parser.parse_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)

    extract_patches(args, config, method=config['method'])
    

if __name__ == '__main__':
    main()