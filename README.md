# Experiments comparing LANET with DNNTSP, SFCNTSP, TCMBN models

To train the models, type the required parameters into utils/config.json and utils/models_configs.json, and run the train.sh script from the root of the directory. You can use the evaluate.sh script in the same way.

If you want to train several models at once, or do it with different sids, you can pass a list with values instead of values to the fields "seed", "model_name", "dataset_name". Then each model will be trained on each dataset specified in the list with dataset names, and with sids specified in the list of sids. 

The experiment uses data in the format from the TCMBN model code. 

After each run of train.sh and evaluate.sh, the configuration is added to the end of the utils/history.txt file

## Adding new models

1) create a folder with the name of your model in the models folder.
2) This folder must contain three files:
- model_creation.py with the create_model function
- prepare_dataloader.py with the function prepare_dataloader
- train_eval_routine.py with functions train_epoch, eval_epoch, evaluate

The function signatures (accepted arguments and returned objects) should be the same as already prepared examples of other models.

Pay attention to the evaluate function. It should save confidence scores and ground truth labels using the save_to_csv function. You can see examples of the format in the code of other models in the train_eval_routine.py file.

3) In models_configs.json add a dictionary with the configuration and the required dataset for your model.
In Main.py and evaluate.py the configuration file is an object called opt. 
It is passed to all the functions in the 3 files described above. To access any characteristic of the "config file", you can do so by referring to an attribute with the same name, e.g. opt.batch_size

When loading a config file, config.json is loaded first, then the dictionary from models_configs.json corresponding to the model and dataset, if there are configuration parameters with the same name in config.json and models_configs.json, then the ones specified in models_configs.json will be used. So, for example, you can store a specific learning rate for a specific model on a specific dataset.

## Adding datasets

The data is stored in the tcmbn_data folder. You can see an example of data preprocessing in preprocess.ipynb


## Datasets

You can find preprocessed datasets here: https://drive.google.com/file/d/1YBqUepwbpDVposLuaKpVE8lJd682S8fE/view?usp=sharing

## Other directories
The saved_models folder stores serialised model objects after training
In the folder model_pred_and_gt the predictions of models on test and valid splits are saved for further analysis by other scripts.



