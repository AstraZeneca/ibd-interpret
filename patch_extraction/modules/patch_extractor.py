from histolab.slide import Slide
from histolab.tiler import RandomTiler
from histolab.tiler import GridTiler
from histolab.masks import TissueMask, BinaryMask, BiggestTissueBoxMask
from histolab.filters.image_filters import (
    ApplyMaskImage,
    GreenPenFilter,
    Invert,
    OtsuThreshold,
    RgbToGrayscale,
)
from histolab.util import rectangle_to_mask
from histolab.types import CP
from histolab.filters.morphological_filters import RemoveSmallHoles, RemoveSmallObjects
import os
from PIL import Image
import numpy as np

class PatchExtractor:
    """
    Class for various extraction methods
    """

    def __init__(self, wsi_path, processed_path, patch_size, level, seed, alpha, check_tissue, tissue_percent, prefix, suffix):
        self.wsi_path = wsi_path
        self.processed_path=processed_path
        self.patch_size = patch_size
        self.level=level
        self.seed=seed
        self.alpha=alpha
        self.check_tissue=check_tissue # default
        self.tissue_percent=tissue_percent # default
        self.prefix=prefix # save tiles in the "random" subdirectory of slide's processed_path
        self.suffix=suffix # default

    def random_extract(self, n_tiles):
        """
        Randomly extract patches from WSI

        Args:
            n_tiles (int): number of patches to extract from WSI
        """

        slide = Slide(self.wsi_path, processed_path=self.processed_path)

        random_tiler = RandomTiler(
                tile_size=self.patch_size,
                level=self.level,
                n_tiles=n_tiles,
                seed=self.seed,
                check_tissue=self.check_tissue, # default
                tissue_percent=self.tissue_percent, # default
                prefix=self.prefix, # save tiles in the "random" subdirectory of slide's processed_path
                suffix=self.suffix # default
            )
        
        random_tiler.locate_tiles(
            slide=slide,
            scale_factor=24,  # default
            alpha=self.alpha,  # default
            outline="red",  # default
        )

        random_tiler.extract(slide)


    def HALO_grid_extract(self, qc_path, pixel_overlap=0, get_mask=False, get_patch_locations=False, save_meta=None, save_patches=False):
        """
        Extract patches in a grid using a QC mask generated by HALO

        Args:
            pixel_overlap (int): number of overlapping pixels between patches, default 0
            get_mask (bool): if True, save an thumbnail of the tissue mask, default False
            get_patch_locations (str): if True, save a thumbnail with the extracted patches shown in red, default False
            save_meta (str): path to save masks and patch locations to
            save_patches (bool): if True, save patches
            qc_path (str): path to HALO QC results

        """

        # open the slide
        slide = Slide(self.wsi_path, processed_path=self.processed_path)

        # check if the slide can be opened
        try:
            thumbnail = slide.thumbnail
        except:
            print("too big - skipping")
            return
        
        qc_path = os.path.join( qc_path, self.wsi_path.split("/")[-1].split(".")[0] + ".tif")

        if os.path.exists(qc_path):
            qc = Image.open( qc_path )
            #qc = qc.resize((np.asarray(thumbnail).shape[1], np.asarray(thumbnail).shape[0]))
            #qc.save('qc_original.png') # used for debugging

            image_file = qc.convert('L')
            #image_file.save("qc_grey.png") # used for debugging
            # Threshold - hard coded 149 is the green region
            image_file = image_file.point( lambda p: 255 if p == 149 else 0 )
            # To mono
            image_file = image_file.convert('1')
            #image_file.save('qc_binary.png') # used for debugging
            
            # define custom binary mask based on HALO
            class HALO_mask(BinaryMask):
                def _mask(self, slide):
                    my_mask = np.asarray(image_file)
                    return my_mask

            mask = HALO_mask()

            filename = self.wsi_path.split("/")[-1].split(".")[0]

            # if get mask is true, save tissue mask
            if get_mask and save_meta is not None:
                print("saving tissue mask...")

                if not os.path.exists( save_meta ):
                    os.mkdir(save_meta)

                if not os.path.exists( os.path.join(save_meta, filename) ):
                    os.mkdir( os.path.join(save_meta, filename) )

                slide_masked = slide.locate_mask(mask)
                slide_masked.save( os.path.join(save_meta, filename, filename + "_tissue_mask.png"), format="png" )
            
            # grid tiler object
            grid_tiler = GridTiler(
                    tile_size=self.patch_size,
                    level=self.level,
                    check_tissue=self.check_tissue, # default
                    tissue_percent=self.tissue_percent,
                    pixel_overlap=pixel_overlap, # default
                    prefix=self.prefix, # save tiles in the "random" subdirectory of slide's processed_path
                    suffix=self.suffix # default
                )
            
            # if get patch locations is true, save patches on thumbnail
            if get_patch_locations and save_meta is not None:
                print("saving patch locations...")

                if not os.path.exists( save_meta ):
                    os.mkdir(save_meta)

                if not os.path.exists(os.path.join(save_meta, filename)):
                    os.mkdir( os.path.join(save_meta, filename) )

                tiles = grid_tiler.locate_tiles(
                    slide=slide,
                    scale_factor=32,  # default
                    alpha=self.alpha,  # default
                    outline="red",  # default
                    extraction_mask=mask
                )

                tiles.save( os.path.join(save_meta, filename, filename + "_patches.png"), format="png" )

            # extract patches
            if save_patches:
                print("extracing patches...")
                grid_tiler.extract(slide, mask)

        else:
            print("no QC result found")
            return

    def grid_extract(self, pixel_overlap=0, get_mask=False, get_patch_locations=False, save_meta=None, save_patches=False):
        
        """
        Extract patches in a grid format from WSIs - main method

        Args:
            pixel_overlap (int): number of overlapping pixels between patches, default 0
            get_mask (bool): if True, save an thumbnail of the tissue mask, default False
            get_patch_locations (str): if True, save a thumbnail with the extracted patches shown in red, default False
            save_meta (str): path to save masks and patch locations to
            save_patches (bool): if True, save patches
        
        """


        
        # open the slide
        slide = Slide(self.wsi_path, processed_path=self.processed_path)

        # check if the slide can be opened
        try:
            thumbnail = slide.thumbnail
        except:
            print("too big - skipping")
            return

        pixels = thumbnail.load() # create the pixel map


        # histolab sometimes identifies the background/unscanned boundary
        # as the tissue/background boundary
        # to address this - white pixels (unscanned) are changed to the average value of background
        for i in range(thumbnail.size[0]): # for every pixel:
            for j in range(thumbnail.size[1]):
                if pixels[i,j] == (255, 255, 255):
                    # if white, change to background colour
                    pixels[i,j] = (219, 218 ,216)
        
        # define tissue mask
        # edit this to change the way that the tissue is identified
        # for example, if the WSIs have many small regions then not using RemoveSmallObjects() might be beneficial
        mask = TissueMask(
            RgbToGrayscale(),
            OtsuThreshold(),
            ApplyMaskImage(thumbnail),
            GreenPenFilter(),
            RgbToGrayscale(),
            Invert(),
            OtsuThreshold(),
            RemoveSmallHoles(),
            #RemoveSmallObjects(),
        )

        class MyCustomMask(BinaryMask):
            def _mask(self, slide):
                thumb = slide.thumbnail
                my_mask = rectangle_to_mask(thumb.size, CP(55, 12, thumb.size[1], thumb.size[0]))
                return my_mask
        
        if self.check_tissue == False:
            mask = MyCustomMask()

        filename = self.wsi_path.split("/")[-1].split(".")[0]

        # if get mask is true, save tissue mask
        if get_mask and save_meta is not None:
            print("saving tissue mask...")

            if not os.path.exists( save_meta ):
                os.mkdir(save_meta)

            if not os.path.exists( os.path.join(save_meta, filename) ):
                os.mkdir( os.path.join(save_meta, filename) )

            slide_masked = slide.locate_mask(mask)
            slide_masked.save( os.path.join(save_meta, filename, filename + "_tissue_mask.png"), format="png" )
        
        # grid tiler object
        grid_tiler = GridTiler(
                tile_size=self.patch_size,
                level=self.level,
                check_tissue=self.check_tissue, # default
                tissue_percent=self.tissue_percent,
                pixel_overlap=pixel_overlap, # default
                prefix=self.prefix, # save tiles in the "random" subdirectory of slide's processed_path
                suffix=self.suffix # default
            )
        
        # if get patch locations is true, save patches on thumbnail
        if get_patch_locations and save_meta is not None:
            print("saving patch locations...")

            if not os.path.exists( save_meta ):
                os.mkdir(save_meta)

            if not os.path.exists(os.path.join(save_meta, filename)):
                os.mkdir( os.path.join(save_meta, filename) )

            tiles = grid_tiler.locate_tiles(
                slide=slide,
                scale_factor=32,  # default
                alpha=self.alpha,  # default
                outline="red",  # default
                extraction_mask=mask
            )

            tiles.save( os.path.join(save_meta, filename, filename + "_patches.png"), format="png" )

        # extract patches
        if save_patches:
            print("extracing patches...")
            grid_tiler.extract(slide, mask)