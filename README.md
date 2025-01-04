# Arabic Named Entity Recognition (NER)

This repository contains the implementation of a system for Arabic Named Entity Recognition (NER). The project uses machine learning and deep learning techniques to identify named entities in Arabic text, such as organizations, persons, locations, and more. The dataset is sourced from [Wojood](https://github.com/qossayrida/ArabicNamedEntityRecognition/tree/main), with a custom mapping of entity labels.

## Dataset

The dataset from Wojood has been adapted to identify the following named entities:

| **Entity** | **Description** | **Mapped From** |
|------------|-----------------|-----------------|
| **ORG**    | Organizations such as companies or institutions. | ORG |
| **TIME**   | Time expressions (e.g., specific times or dates). | TIME, DATE |
| **LOC**    | Locations, including geopolitical and geographical locations. | GPE, LOC |
| **MON**    | Monetary values and currencies. | CURR, MONEY |
| **PER**    | Persons or groups of people. | NORP, PERS |
| **EVE**    | Named events. | EVENT |
| **NUM**    | Numerical values (e.g., percentages, quantities). | PERCENT, QUANTITY, CARDINAL |
| **LAN**    | Languages. | LANGUAGE |

### Excluded Entities
Entities such as `WEBSITE`, `OCC` (occupation), `FAC` (facilities), `PRODUCT`, `LAW`, `UNIT`, and `ORDINAL` are excluded from this system.



## Repository Structure

The repository is organized into the following directories:

- **backend**: Contains Python scripts for backend processing and model integration.
  - `assemblage.py`: Combines and preprocesses data.
  - `main.py`: The main script for running the application.

- **data**: Includes raw and preprocessed datasets.
  - `entity.txt`: Entity definitions and mappings.
  - `train.txt`, `val.txt`, `test.txt`: Raw training, validation, and test datasets.
  - `*_cleaned.txt`: Cleaned and preprocessed versions of the datasets.

- **gui**: Contains files for the graphical user interface (GUI).
  - `index.html`: HTML structure for the web interface.
  - `script.js`: JavaScript for interactive functionalities.
  - `styles.css`: CSS for styling the interface.

- **jupyter**: Jupyter notebooks for data analysis and model development.
  - `crf_model.ipynb`: Implements the Conditional Random Field (CRF) model.
  - `data_analysis.ipynb`: Performs exploratory data analysis (EDA) on the dataset.
  - `decision_tree_model.ipynb`: Decision tree-based approach for NER.
  - `naive_bayes_model.ipynb`: Naive Bayes-based approach for NER.
  - `rnn_model.ipynb`: RNN-based deep learning model for NER.



## How to Use

1. **Data Preparation**: Ensure the datasets in the `data` directory are properly formatted and cleaned.
2. **Model Training**: Use the Jupyter notebooks in the `jupyter` directory to train or fine-tune models.
3. **Run Backend**: Execute the backend scripts in the `backend` directory to process data and integrate models.
4. **GUI**: Use the files in the `gui` directory to interact with the system via a web interface.



## Key Features

- Supports multiple NER models: CRF, Naive Bayes, Decision Tree, and RNN.
- Handles Arabic text with preprocessing and normalization.
- Custom entity mapping for tailored recognition tasks.



## Requirements

- Python 3.x
- TensorFlow or Keras for deep learning models
- Scikit-learn for classical machine learning models
- Flask or a similar framework for backend processing
- Standard web technologies (HTML, CSS, JavaScript) for the GUI



## Sample Run

Here is a sample run of the Arabic NER system:

<p align="center">
  <img src="https://github.com/user-attachments/assets/b5a68d88-27a0-4bec-8fb8-0b579b4fc2e6" alt="Sample Run" width="600">
</p>




## Acknowledgments

- Dataset from [Wojood](https://sina.birzeit.edu/wojood/)
- Inspired by research on Arabic NLP and NER.


## 🔗 Links

[![facebook](https://img.shields.io/badge/facebook-0077B5?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/qossay.rida?mibextid=2JQ9oc)

[![Whatsapp](https://img.shields.io/badge/Whatsapp-25D366?style=for-the-badge&logo=Whatsapp&logoColor=white)](https://wa.me/+972598592423)

[![linkedin](https://img.shields.io/badge/linkedin-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/qossay-rida-3aa3b81a1?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app )

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/qossayrida)
