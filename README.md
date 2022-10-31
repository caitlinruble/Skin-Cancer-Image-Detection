# Skin-Cancer-Image-Detection
Can deep learning help the everyday user know whether a mole is normal, or whether they should really get that checked out? (Spoiler Alert--yes, it most certainly can!)

### Repository Key Contents:
- [Model Metrics File](https://github.com/caitlinruble/Skin-Cancer-Image-Detection/blob/8886d25c4b03304b14c58e913b0fb9dfe9559f90/Model%20Metrics.md)
- [requirements.txt](requirements.txt)
- [Full Report](https://github.com/caitlinruble/Skin-Cancer-Image-Detection/blob/dd7cb50a9c2dbf55f4c4cd90c1f36037c6e671da/Reports/Melanoma%20Image%20Detection_Final%20Report.pdf)
- [Slide Deck Presentation to Executive Audience](Reports/Melanoma_AI_presentation_slides.pdf)
- [Final Model Training Notebook](melanoma-detection-using-fastai-on-resnet34-base.ipynb)
- [Model Development and Experimentation Notebook](https://github.com/caitlinruble/Skin-Cancer-Image-Detection/blob/dd7cb50a9c2dbf55f4c4cd90c1f36037c6e671da/Development%20Notebooks/melanoma-detection-using-fastai-hyperparameter-experiments.ipynb)
- [Exploratory Data Analysis Notebook]()


### Abstract:

Melanoma is a deadly form of skin cancer that is highly curable when caught early. A deep-learning computer vision tool was developed to identify melanoma from images of skin lesions. This tool was trained on the SIIM-ISIC Melanoma Challenge dataset available on Kaggle using the FastAI library to interface with the ResNet34 pre-trained Convolutional Neural Network (CNN). The final model selected uses the average probability across 5 cross-validation folds to classify whether an image shows a malignant melanoma, or a benign skin lesion. In private validation testing, the ensembled model had a 90% chance of distinguishing between the positive and negative classes (AUROC = 0.9) and correctly classified 83% of the melanoma images (recall = 0.83). In further validation testing through submission to Kaggle, the model had an 85% chance of distinguishing between the malignant and benign classes (AUROC = 0.8531). A proposed use case for this model is as a screening tool for home use; early, easy, and skillful screening with artificial intelligence tools can effectively get more patients in the door for early treatment of melanoma. Getting more patients in the door helps melanoma patients by increasing their survival rates and decreasing the cost and intensity of treatment, hospital systems by reducing the number of patients in need of in-hospital surgery and chemotherapy and thereby reducing system strain, and health insurance companies by significantly reducing the cost of treating the same condition when caught early vs. late.

### Results:

The ensembled CNN learner model score an AUROC of 0.85 on the "private" Kaggle test image set containing 7,687 unseen images, indicating that the model has an 85% chance of distinguishing between melanoma and benign skin lesions. In the "public" test set of 3,295 images, it scored AUROC = 0.87; the fact that this score was close to the "private" score supports the conclusion that the model performance shows stability across different test sets. The weighted average AUROC across all 10,982 test images was 0.858.
<img width="387" alt="Screen Shot 2022-10-31 at 1 59 07 PM" src="https://user-images.githubusercontent.com/96548036/199077019-95f6c09a-54da-47ac-9d39-568503192e7b.png">


### My approach: 

This computer vision problem called for a deep learning approach, and the use of a pretrained convolutional neural network (CNN) to be fine-tuned on the skin lesion data set was employed. The FastAI library coupled with the ResNet34 pretrained CNN was the selected approach. FastAI is a high-level framework built on top of PyTorch. The classes in the training images were balanced by down-sampling the benign images, and the resulting training subset was further split into a training and internal-validation set. The training data was manually split into 5 cross-validation folds, and the ResNet34 learner was trained separately 5 times over this cross-validation split. In each training, the ResNet34 model was fine-tuned over 15 epochs, optimizing for the binary RocAucScore metric, and the best epoch was saved as the best model for that CV fold. The 5 find-tuned CNN learners were ensembled, and the mean prediction value given by each was taken as the final prediction values for the test set.

### The Dataset:

The ["SIIM-ISIC Melanoma Classification"](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data) dataset on Kaggle contains 33,126 labeled images and metadata for the 2,056 patients the images are taken from as training data. It contains an additional 10,982 unlabeled images with associated metadata as a test set. The images are available as DICOM files, JPG files, and TFRecord files. The metadata is available within the DICOM files and within the .csv files “train.csv” and “test.csv”. In addition to this dataset, the [SIIM ISIC - 224x224 image set](https://www.kaggle.com/datasets/arroqc/siic-isic-224x224-images) uploaded to Kaggle by user Arnaud Roussel were leveraged for model training. This dataset contained all the same images as the primary competition dataset, but had been resized to a standard 224x224 image size and saved as .png files. Using these resized images allowed for efficient model training relative to using FastAI to manually resize each image before being loaded for model training.

### Suggested Uses:

This model creates value for several main entities: patients, healthcare providers, and health insurance providers.

<img width="781" alt="Screen Shot 2022-10-31 at 2 05 56 PM" src="https://user-images.githubusercontent.com/96548036/199078278-e5b76cc7-eafd-4c19-bacc-87ea483a9816.png">

My suggested deployment of this model is as a web or mobile application a person can navigate to, upload an image of their mole, and receive a screening result. If their result is positive, the patient would be directed to educational resources about the importance of early screening and treatment and would be directed to make an appointment with a specialist for medical confirmation. This process is summarized in Figure 9, below. A health insurance company could build this out into a specialize app for their members, potentially offering “wellness rewards” for participating in the screening and following up on any recommendations, tracking individual moles over time, and directly linking to providers who are covered in-network for ease of member follow-through. This tool could also be used by primary care physicians, dermatologists, hospital systems, or directly by curious people interested in being proactive about their healthcare.

<img width="903" alt="Screen Shot 2022-10-31 at 2 06 21 PM" src="https://user-images.githubusercontent.com/96548036/199078353-b33dc68c-09fe-4d6f-b847-46cb53491d64.png">


### Limitations and Future Work

1. One of the major limitations of this dataset and therefore model is the lack of melanin diversity in the patient images. All cases of melanoma are from non-melanated (i.e. “white”) people, as well as all of the observed benign images. While it’s true that the vast majority of melanoma cases are in white people, BIPOC are also afflicted by the disease and often have a worse prognosis due to being diagnosed once the disease is in a more progressed state. Care must be taken to ensure patients with melanated skin do not rely on this tool for medical screening. Unfortunately, medical research involving the use of imaging and light often centers around people with white skin, contributing to racial and ethnic group injustice in medicine. We have to do better by including representative samples in our research and validation!

2. The current model only takes into account the image data, excluding patient metadata that could strengthen its predictive power. A future iteration could blend image and tabular metadata in the deep learning training process.

3. In order for this model to become a useful tool, it needs to be deployed as a web app or developed into an app interface. We should keep in mind the experience of a person using the app, and strive to be transparent, informative and interpretable to the lay-person.

4. A final suggestion would be adding a memory/temporal element to model. Images of the same skin lesion, taken over time, could give valuable insight into any changes occurring and lend data support toward or against a melanoma categorization.
