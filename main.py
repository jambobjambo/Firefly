import DataStorage.training_data_convert as DataConv

#Step 1 | Encode Training Data
def generate_training_data():
    print("Encoding Files")
    DataConv.main()
    print("Encoding Complete")

#Step 2 | Import Training Data
train_x, train_y, test_x, test_y = DataConv.create_feature_sets_and_labels()

#Step 3 | Feed data to machine learning model


gen_new_data = True
if __name__ == '__main__':
    if gen_new_data == True:
        generate_training_data()
