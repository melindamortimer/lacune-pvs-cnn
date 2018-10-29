# lacune-pvs-cnn

Model development for the identification of lacunes in MRI.

**Thesis_Docs/**: contains tex, images, and bibliography for the thesis. Does not include the pdf.

**Presentation/**: contains presentation tex, single image, and edited version of bib (shorter names).

**NNet_Scripts/**: Model development files:

1. *1_lacune_cnn_1.R*: 2D CNN code. Requires training, validation, and testing set files. Also assesses model confusion: FP, FN, TP, TN.
2. *1_lacune_cnn_2.R*: 3D CNN code. Unused.
3. *2_ImportMRI.R*: Initial sampling of data. INCORRECT.
4. *3_plots.R*: Generated plots for thesis. Training and validation accuracies.
5. *4_eval.R*: Attempt to apply model to whole scan. UNFINISHED.
6. *5_resample.R*: CORRECTED sample generation.
7. *6_appendix_code.R*: Simplified 2D CNN code for inclusion in thesis appendix.
8. *7_test_small_patch.R*: Attempt to apply model to part of a scan. UNFINISHED.
