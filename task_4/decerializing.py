import dill

# IT WORKS
def main():
    file_name = '/home/jollyreap/old_linux/ML_Engineer/Cognizant_intership/task_4/model/fr_stock_predictor.pkl'
    with open(file_name, 'rb') as file:
        model = dill.load(file)

    print(model['metadata'])


if __name__ == "__main__":
    main()