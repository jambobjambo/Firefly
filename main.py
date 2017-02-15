import DataStorage.training_data_convert as DataConv

#Step 1 | Import Training Data
def generate_training_data():
    print("Encoding Files")
    DataConv.main()
    print("Encoding Complete")

#Step 2 | Feed data to machine learning model
def create_neural_net():
    

gen_new_data = True
if __name__ == '__main__':
    if gen_new_data == True:
        generate_training_data()
