<h1 align="center"> Vivim: a Video Vision Mamba for Medical Video Segmentation </h1>

## Overview
Vivim, originally developed for binary video object segmentation, offers an alternative to Vision Transformer–based models. Its core architecture uses State Space Models, specifically Temporal Mamba Blocks, in place of the standard Transformer attention layer. In this project, we extend Vivim to multiclass segmentation and optimize it for accurate, efficient ovarian tumor segmentation.

## Our Approach
This project aims to improve the original company model (U-Net) by leveraging temporal-aware video object segmentation, rather than treating each frame independently. Our main contributions include:

1. **Domain-Specific Data Preprocessing**
- **Standard augmentations**: Random cropping, flipping, rotation, and Gaussian blur
- **Fan cropping**: We cropped the frames by following the fan beam binary segmentation mask, in order to exclude the noise from outside the beam
- **Frame selection strategy**: Introduction of max_numerosity parameter to limit frames per clinical case. Two mehtods to select frames has been explored: 
    - Method 1 : Equispaced frame selection, fixed throughout training
    - Method 2: Random frame selection at each epoch

In this case, since Vivim works with clips of frames, so max_numerosity is applied on clips and not on single frames.
Empirical results show Method 1 provides better accuracy and efficiency

2. **Training Framework**
- **Segmentation types**: Support for both binary and multiclass segmentation
- **Classes**: Background, solid, and non-solid (for multiclass)
- **Loss function**: We tried a variety of losses, the one that we settled with is a weighted combination of class-balanced focal loss and Tversky loss
- **Optimizer**: AdamW with CosineAnnealingLR scheduler
- **Training Strategy**: 5-fold cross-validation for hyperparameter optimization, followed by a full retraining with no validation.
- **Model Architecture**: We expand the model to support multiclass segmentation. We added various Dropout layers to improve model's generalization

3. **Inference Pipeline**
- **Evaluation**: Evaluation metrics and confusion matrices are computed

4. **Results and Future Improvement**
- **Results**: Vivim outperforms the exsisting model in both segmentation accuracy and inference speed, providing an excellent segmentation model choice. The full results will be shared with the pubblication of the thesis.
- **Future Improvement**: Explore different clip length (we have just tested with 3 and 5), and further hyperparameters optimization are needed to reach the full model potential.

## Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate vivim``


 Also install the requirements in:

 ``requirements.txt``


 ## How to run:
 
**Step1:** Create the 5 folds that will be used in the training phase running the following script:
``python multiclass_StratKFold.py ``

 **Step2:** Run the multiclass training and validation by:
 
``python multiclass_training_folds.py -val_freq 10 -image_size 256 -clip_length 5 -train_bs 3 -epochs 100 -num_workers 4 -cv_group Vivim_with_recall_loss -num_folds 3 -max_num 3 -num_classes 3``

**Step3:** In order to run the final training (using all folds as training, so no validation) do:

``python final_multiclass_training.py -val_freq 100 -image_size 256 -clip_length 5 -train_bs 3 -epochs 100 -num_workers 4 -cv_group Vivim_final_multiclass_training -max_num 3 -num_classes 3``

**Step4:** Inference script:

``inference.py --ckpt multiclass_checkpoints_final/best_Twersky_loss.ckpt --wandb_project vivim_inf_try``

**Notes:**  
-The scripts that terminate with `_dyn` are the ones that dynamically change the clips to consider (as a consequence of max_numerosity) at each training step (performance is better with the standard approach, by taking equispaced clips).    
-The dataset is preprocessed in `Multiclass_Data.py`.


 ## Original Vivim:

 repo: https://github.com/scott-yjyang/Vivim?tab=readme-ov-file  
 paper: https://arxiv.org/abs/2401.14168










