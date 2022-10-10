# Skin-Cancer-Image-Detection
Can deep learning help the everyday user know whether a mole is normal, or whether they should really get that checked out? (Spoiler Alert--yes, it most certainly can!)

### Project Outcomes:
  1. A functional web app where you can upload an image of a mole and my model will predict the likelihood that your mole is malignant.
  2. KPIs of the final selected model
  3. A submission to the [SIIM-ISIC Melanoma Classification Competition on Kaggle.]<https://www.kaggle.com/competitions/siim-isic-melanoma-classification>
  
### My approach: 
I elected to use the FastAI library, which is built on top of PyTorch and makes it simple to leverage pre-trained models to acheive state-of-the-art results. 

### The Dataset
I am utilizing the [SIIM-ISIC Image Dataset]<https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data> from Kaggle, which includes over 33,000 labeled training images and a validation set of an additional nearly 11000 images. These image files are available in several different formats: DICOM, JPEG and TFRecord. 

### Exploratory Data Analysis and Data Cleaning


### Modeling Experiments


### Final Model Selection


### Challenges and Side-quests
  1. One of the biggest challenges with using this dataset was how **big** the dataset was! I really wanted to use Paperspace Gradient for this project, because I happen to have quite a few credits with them. However, there is no "easy" way to mount the 35 GB of JPEG image data from Kaggle and into Paperspace. Their GUI limits uploads to 5000 files at a time, and they suggest using their CLI tool to upload larger datasets. I tried this every which way, following their online guides and even contacting support, but with no resolution. The API timed out every time I tried to upload the files directly through the CLI. I wrote a python script to move the images into batches of 5000 to try through the GUI like that, to no avail. So then [I wrote a python script to "unbatch" the images]<https://github.com/caitlinruble/Skin-Cancer-Image-Detection/blob/00cb45f78bdb59c28f8402cba81bfc51cad8acc1/Python%20scripts/revert_jpegs_to_original_directory>! Ultimately, after a week of trying different ways of getting these images into Paperspace to use their cloud-based GPUs, I "gave up" and reverted to simply using Kaggle notebooks and the 15GB of GPU available to me there.

