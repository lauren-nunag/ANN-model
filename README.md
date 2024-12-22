Here's a breakdown of the code so you don't lose your mind :3 

createSynthesizer - creates the CTGAN synthesizer that is now being used, loads the dataset from file (I did this really shittily), creates metadata which is needed for model & data synthesis and saves into .json
we then verify the metadata, once verified we fit the model to the data and then we run library's quality report and then create a scatter plot of the distrbution of the real and synthesized data. Saves the model in a .pkl to be used later

dataSynthesizer - synthesizes data from trained model :o . synthesize(rows) call this function to create a .xslx file of synthesized data w/ user specified number of rows. This function also calls cleanSheet(excel_file)
which takes an excel file and removes any rows with data missing. 
splitSheet(excel_file) will take an excel file (made specific for a file where the first column is the input and the other 3 columns are the outputs) and split the first 70% of rows into train, 20% for validation, and 10% for application
Will turn file into 6 sheets, 3 input(TVA) & 3 output(TVA) TVA - train, validate, apply
normalize_and_save_params(excel_file) - **work in progress** end goal should be after an excel file is cleaned but before it is split it should be normalized. The function will find the max of every column and divide each row in the column by the max so then the range is basically between [0,1]
This function should technically work but idk

You don't need to work with the original csv files at this point (which is why it is in the .gitignore) as the .pkl should suffice
