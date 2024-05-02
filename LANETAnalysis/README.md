How to use. Instructions

Extracting data from the model is left to the user, as the models code and internal data format is usually different.

You must inherit the ProcessedDataset class from the utils/DataClass.py file and be sure to override the abstract methods of the parent class in it, as well as the model_name and dataset_name fields. Each method should return an np.ndarray of size (num_of_samples, num_of_labels)

You pass the object of your given subclass as input to the DataClass class initialiser.

Further, you can read an example of using the resulting object in the notebook example.ipynb
