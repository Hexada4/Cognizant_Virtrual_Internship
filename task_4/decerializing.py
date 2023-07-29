import dill

# IT WORKS
def main():
    file_name = 'your_path/task_4/model/fr_stock_predictor.pkl'
    with open(file_name, 'rb') as file:
        model = dill.load(file)

    print(model['metadata'])


if __name__ == "__main__":
    main()
