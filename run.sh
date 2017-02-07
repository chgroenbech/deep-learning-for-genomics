#!/usr/bin/env bash

DATA_NAME="GSE63472_P14Retina_merged_digital_expression"
CLUSTER_NAME="retina_clusteridentities"
LATENT_SIZES=50
HIDDEN_STRUCTURE=500
FILTERING_METHOD="Macosko"
# FILTERING_METHOD="clusters 2 25 36 37"
# SPLITTING_METHOD="Macosko"
SPLITTING_METHOD="random"
SPLITTING_FRACTION=0.8
FEATURE_SELECTION="high_variance"
FEATURE_SIZE=5000
# RECONSTRUCTION_DISTRIBUTIONS="bernoulli"
RECONSTRUCTION_DISTRIBUTIONS="poisson"
# RECONSTRUCTION_DISTRIBUTIONS="negative_binomial"
# RECONSTRUCTION_DISTRIBUTIONS="zero_inflated_poisson"
# RECONSTRUCTION_DISTRIBUTIONS="zero_inflated_negative_binomial"
NUMBERS_OF_RECONSTRUCTION_CLASSES=0
# NUMBERS_OF_RECONSTRUCTION_CLASSES=6
# NUMBERS_OF_EPOCHS="5"
# NUMBERS_OF_EPOCHS="10 20 50"
# NUMBERS_OF_EPOCHS="50"
NUMBERS_OF_EPOCHS="100"
BATCH_SIZE=100
# LEARNING_RATE=1e-3
# LEARNING_RATE=1e-4
LEARNING_RATE=1e-5
NUMBER_OF_WARMUP_EPOCHS="0"

./src/main.py --data-name $DATA_NAME --cluster-name $CLUSTER_NAME \
              --latent-sizes $LATENT_SIZES \
              --hidden-structure $HIDDEN_STRUCTURE \
              --splitting-method $SPLITTING_METHOD \
              --splitting-fraction $SPLITTING_FRACTION \
              --filtering-method $FILTERING_METHOD \
              --feature-selection $FEATURE_SELECTION \
              --feature-size $FEATURE_SIZE \
              --reconstruction-distributions $RECONSTRUCTION_DISTRIBUTIONS \
              --numbers-of-reconstruction-classes $NUMBERS_OF_RECONSTRUCTION_CLASSES \
              --numbers-of-epochs $NUMBERS_OF_EPOCHS \
              --number-of-warm-up-epochs $NUMBER_OF_WARMUP_EPOCHS \
              --batch-size $BATCH_SIZE \
              --learning-rate $LEARNING_RATE \
              $1
              # --use-batch-norm \
