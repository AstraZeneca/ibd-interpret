# Patch Extraction Code

Code adapted from Histolab: https://github.com/histolab/histolab 

## 1) Setup Environment

Set up an environment to run the patch extraction code.

```
conda create --name extractor python=3.8.13
conda activate extractor
```

Once the environment is created, install Openslide and then the Python packages from requirements.txt

```
conda install -c conda-forge openslide
pip3 install -r requirements.txt
```

## 2) Organise WSIs

Simply store all WSIs in a folder e.g.

```
XXXXX.tiff
XXXXX.tiff
XXXXX.tiff
.
.
.
```

## 3) Run Patch Extraction

The python command has the following structure:

```
python modules/extractor.py \
--meta_path="path/to/metadata/save/directory" \
--wsi_path="path/to/WSIs" \
--processed_path="path/to/where/to/save/patches" \
--config_path="config.yaml" \
--QC_path="path/to/QC/results"
```

`--meta_path` is where tissue masks and tile locations will be saved, `--processed_path` is where patches will be saved. `--QC_path` can be used to specify QC data for patching. Use the `config.yaml` to specify the parameters of patch extraction. See details below.


### Config.yaml

| Parameter | Description |
| --------- | ----------- |
| method | Patch extraction method, defaults to grid_extract |
| patch_size | Size of patches (patch_size x patch_size) |
| level | The level of magnification to access (e.g. 0 is the highest magnification) |
| check_tissue | If True, the tissue will be segmented and patches will be located only in tissue region |
| tissue_percent | Specify the percent of tissue needed in patch to be saved |
| suffix | Extension for patches e.g. .png |
| pixel_overlap | Percentage overlap of patches e.g. 0.2 = 20% overlap |
| alpha | The level of alpha in the patch location overlay - 255 will show no tissue, 0 will show no patch grid. |
| get_mask | If True, save the tissue mask as a thumbnail |
| get_patch_locations | If True, save the patch locations as a thumbnail |
| save_patches | If True, save the patches |

