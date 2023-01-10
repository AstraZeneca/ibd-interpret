# Hierarchical Image Pyramid Transformer (HIPT) Code

Adapted from https://github.com/mahmoodlab/HIPT 


# Setup Environment

Set up an environment to run HIPT.

```
conda create --name HIPT python=3.8
conda activate HIPT

pip install -r requirements
```


# Finetuning HIPT

## 0) Folder organisation

Folders should be organised as:

- WSI_root
    - TASK_subtype
        - slide_1
        - slide_2 etc. (where slide_x are the folders containing patches that are extracted by patch extraction)
        - extracted_mag20x_patch4096_fp
            - vits_tcga_pancancer_dino_pt_patch_features_384
                - slide_1.pt
                - slide_2.pt etc. (these are the embeddings produced for each WSI)

## 1) Generate 4096x4096 patches

For generating 4096x4096 we find that CLAM works better than histolab. Please see the [CLAM Repo](https://github.com/mahmoodlab/CLAM)

## 2) Generate embeddings for the patches using pretrained 2-stage HIPT

Once patches are generated, embeddings for the patches must be generated using a pretrained 2-stage HIPT. To do this run the following command:

```
CUDA_VISIBLE_DEVICES=0,1 python -u generate_region_embeddings.py --patch_dir="WSI_root/TASK_subtype" --feats_dir="WSI_root/TASK_subtype/extracted_mag20x_patch4096_fp/vits_tcga_pancancer_dino_pt_patch_features_384/" > outfiles/generate_embeddings.out
```

## 3) Organise the slide-level labels

In order to specify a new fine tuning experiment, open main.py in 2-Weakly-Supervised-Subtyping and scroll to the bottom of the file where many tasks are defined. Here is an example task:

```python
if args.task == 'tcga_lung_subtype':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = './dataset_csv/tcga_lung_subset.csv.zip',
                            data_dir= os.path.join(args.data_root_dir, study_dir),
                            mode=args.mode,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='oncotree_code',
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            prop=args.prop,
                            ignore=[])
```

Here we can see that the task 'tcga_lung_subtype' has 2 classes and has an input dataframe tcga_lung_subset.csv. The slide level labels are in a column oncotree_code and the mapping of the classes is LUAD:0 and LUSC:1. In order to define a new prediction task, use this format. The dataframe passed into csv_path should contain case_id (patients), slide_id and label columns.

This code is set up to do k-fold cross validation. These are passed in as csv files with columns train, val and test which each store a list of the slides belonging to each set. The csv should be stored in splits/.



## 4) Finetune the last stage of HIPT on the embeddings

Once embeddings have been generated and labels have been organised, run the training script:

```
GPU=0
DATAROOT=../WSI_root
TASK=TASK
CUDA_VISIBLE_DEVICES=$GPU python -u main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0 > ../outfiles/finetune_HIPT.out
```

# End to end training HIPT

The first 2 vision transformers that make up HIPT (ViT:16-256 and ViT:256-4K) can be pretrained hierarchically on unlabeled images. Doing this is comparatively simple to finetuning, but training times are much much longer. All pretraining code is in 1-Hierarchical-Pretraining

## 0) Folder organisation

For pretraining both models, set up a folders like this:

* PRETRAINING_DIR
    * patch_256_pretraining
    * region_4096_pretraining
    * vit256_ckpts
    * vit4k_ckpts

patch_256_pretraining is where the patches for training ViT:16-256 will be stored, region_4096_pretraining is where the embeddings for training ViT:256-4K will be stored.

Checkpoints will be stored in the corresponding ckpts folders.

## 1) Pretraining ViT:16-256

### i) Splitting the 4K patches into 256x256 patches

Since HIPT is hierarchical, ViT:16-256 trains on the 256 256x256 patches that are inside each 4096x4096 region. Therefore, the first step is to split each 4096x4096 region into 256 256x256 patches and save each patch in a pretraining directory. To do this run the following command:

```
python -u parse_4k.py --patch_dir="WSI_root/TASK_subtype" --save_dir="TASK_pretraining/patch_256_pretraining" > outfiles/parse_4k.out
```

### ii) Training ViT:16-256 on the patches

After patches are generated, train ViT:16-256 on the patches by running the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch vit_small --data_path  --output_dir  --epochs 100
```

The above command is for running on 4 GPUs. To run on a different number, change CUDA_VISIBLE_DEVICES and --nproc_per_node to be the same number.

## 2) Pretraining ViT:256-4K

### i) Generate embeddings using trained ViT:16-256

ViT:256-4K trains on the embeddings outputted by ViT:16-256, so the first step is to generate these embeddings using the above trained model. To do this, run the following command:

```
CUDA_VISIBLE_DEVICES=0,1 python -u generate_patch_embeddings.py \
--patch_dir="WSI_root/TASK_subtype" \
--feats_dir="PRETRAINING_DIR/region_4096_pretraining" > outfiles/generate_patch_embeddings.out
```

### ii) Train ViT:256-4K on the generated embeddings

Once embeddings have been generated, train ViT:256-4K using the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 main_dino4k.py --arch vit4k_xs --data_path ../PRETRAINING_DIR/region_4096_pretraining --output_dir ../PRETRAINING_DIR/vit4k_ckpts --epochs 100
```

Adjust GPUs same as above

After pretraining is complete, you can replace ViT:16-256 and ViT:256-4K in the fine tuning scripts above