# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a machine learning classifier trained to predict whether an individual's income exceeds $50,000 per year based on census data. The model was trained using the Adult Census Income dataset and a machine learning pipeline that includes categorical feature encoding and label binarization. 

## Intended Use
This model is intended for educational purposes as part of the Udacity Machine Learning DevOps project. Its purpose is to demonstrate how to train, evaluate, save, and deploy a machine learning model using a reproducible pipeline.

## Training Data
The training data comes from the Adult Census Income dataset. It includes demographic and employment-related features such as workclass, education, marital status, occupation, relationship, race, sex, and native country, along with numerical features. 

## Evaluation Data
The evaluation data comes from the same Adult Census Income dataset and consists of the remaining 20% of the data after the train-test split. This held-out test set was not used during training and was used to evaluate the model’s general performance. 

## Metrics
The model was evaluated using precision, recall, and F1 score. These metrics were chosen because they provide a balanced view of classification performance, especially when class distributions may be uneven. On the test set, the model achieved a precision of 0.7402, a recall of 0.6384, and an F1 score of 0.6856. Model performance was also examined across categorical slices to better understand how performance may vary for different groups.


## Ethical Considerations
This model uses census-based demographic and socioeconomic features, some of which are sensitive, including race and sex. Because of this, the model may learn historical biases present in the data and could produce unfair outcomes for certain groups. Predictions from this model should therefore be interpreted carefully, and fairness analysis should be considered before any real-world use.

## Caveats and Recommendations
This model was developed for a classroom project and is limited by the quality, age, and representativeness of the dataset. Its performance may not generalize well to current populations or to data collected in different contexts. It is recommended that this model only be used for demonstration and learning purposes. Before any production use, the model should undergo additional validation, fairness testing, monitoring, and possible retraining on more current and representative data.