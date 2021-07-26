# Data Labeling

[Reference](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-labeling-projects)

- `Note: Data images or text must be available in an Azure blob datastore`
- `Image data can be files with any of these types: ".jpg", ".jpeg", ".png", ".jpe", ".jfif", ".bmp", ".tif", ".tiff", ".dcm", ".dicom".`
- `Each file is an item to be labeled.`
- `Review the labeled data and export labeled in COCO format or as an Azure Machine Learning dataset.`

## Multi-class versus Mulit-label

- Choose Image Classification **Multi-class** for projects when you want to apply only a**single label** from a set of labels to an image.
- Choose Image Classification **Multi-label** for projects when you want to apply **one or more labels** from a set of labels to an image. For instance, a photo of a dog might be labeled with both dog and daytime.
- Choose Object Identification (Bounding Box) for projects when you want to assign a label and a bounding box to each object within an image.
- Choose Instance Segmentation (Polygon) for projects when you want to assign a label and draw a polygon around each object within an image.

### Aside

- **Multiclass classification** means a classification task with more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Multiclass classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time.

- **Multilabel classification** assigns to each sample a set of target labels. This can be thought of as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A text might be about any of religion, politics, finance or education at the same time or none of these.

## Text labeling is in preview

- Choose Text Classification **Multi-class (Preview)** for projects when you want to apply only a **single label** from a set of labels to each piece of text.
- Choose Text Classification **Multi-label (Preview)** for projects when you want to apply one or more labels from a set of labels to each piece of text.

## Configure incremental refresh
If you plan to add new files to your dataset, use incremental refresh to add these new files your project. When incremental refresh is enabled, the dataset is checked periodically for new images to be added to a project, based on the labeling completion rate. The check for new data stops when the project contains the maximum 500,000 files.

## Specify label classes
On the Label classes page, specify the set of classes to categorize your data. Your labelers' accuracy and speed are affected by their ability to choose among the classes. For instance, instead of spelling out the full genus and species for plants or animals, use a field code or abbreviate the genus.

## Provide Instructions
On the Labeling instructions page, you can add a link to an external site for labeling instructions, or provide instructions in the edit box on the page. Keep the instructions task-oriented and appropriate to the audience. Consider these questions:


## For human labeling

#### Access for labelers
Anyone who has access to your workspace can label data in your project. You can also customize the permissions for labelers so that they can access labeling but not other parts of the workspace or your labeling project. For more details, see Manage access to an Azure Machine Learning workspace, and learn how to create the labeler custom role.

## Use ML-assisted data labeling
The ML-assisted labeling page lets you trigger automatic machine learning models to accelerate labeling tasks. It is only available for image labeling. Medical images (".dcm") are not included in assisted labeling.
#### Phases
- Clustering (for image labeling)
- Prelabeling


