#!/usr/bin/env bash

DATA_NAME="GSE63472_P14Retina_merged_digital_expression"
CLUSTER_NAME="retina_clusteridentities"
LATENT_SIZE=50
HIDDEN_STRUCTURE=500
FILTERING_METHOD="Macosko"
SPLITTING_METHOD="random"
SPLITTING_FRACTION=0.8
FEATURE_SELECTION="high_variance"
FEATURE_SIZE=5000
# RECONSTRUCTION_DISTRIBUTION="negative_binomial"
# RECONSTRUCTION_DISTRIBUTION="bernoulli"
RECONSTRUCTION_DISTRIBUTION="zero_inflated_poisson"
NUMBER_OF_EPOCHS=10
BATCH_SIZE=100
# LEARNING_RATE=1e-4
LEARNING_RATE=1e-3
# LEARNING_RATE=1e-2

./src/main.py --data-name $DATA_NAME --cluster-name $CLUSTER_NAME \
              --latent-size $LATENT_SIZE \
              --hidden-structure $HIDDEN_STRUCTURE \
              --filtering-method $FILTERING_METHOD \
              --splitting-method $SPLITTING_METHOD \
              --splitting-fraction $SPLITTING_FRACTION \
              --feature-selection $FEATURE_SELECTION \
              --feature-size $FEATURE_SIZE \
              --reconstruction-distribution $RECONSTRUCTION_DISTRIBUTION \
              --number-of-epochs $NUMBER_OF_EPOCHS \
              --batch-size $BATCH_SIZE \
              --learning-rate $LEARNING_RATE \
              $1
