## Model Metrics

### Final Model: 
An ensemble of 5 neural networks, each trained on a different fold of the training data. 

### Parameters:
Pre-trained model = ResNet34 \
Monitoring function = RocAucBinary

### Hyperparameters:
Batch transformers = Normalize.from_stats(*imagenet_stats) \
batch size = 8 \
n_epochs = 15 \
call backs = [SaveModelCallback(monitor='roc_auc_score', comp = np.greater, with_opt=True), ReduceLROnPlateau(monitor = 'roc_auc_score', comp = np.greater, patience=2)])

### Performance:

**Kaggle Competition ROCAUC Leaderboard Scores:**

Private AUROC = 0.8531
Public AUROC = 0.8702

**Private Validation on Unseen Test Data:** 

Validation Set Precision: 0.83 \
Validation Set Recall: 0.83 \
Validation Set Accuracy: 0.83 \
Validation Set Average Precision Score: 0.9 \
Validation Set Roc Auc Score: 0.9

**Cross Validation Scores for Best Model:**

mean_roc_auc_CV = 0.885 \
mean_error_rate_CV = 0.201 \
mean_precision_CV = 0.793 \
mean_recall_CV = 0.812 


