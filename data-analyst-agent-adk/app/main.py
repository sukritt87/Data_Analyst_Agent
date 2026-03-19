from tools.dataset_manager import DatasetManager


def main() -> None:
    dataset_manager = DatasetManager("data/sample_data.csv")
    dataset_manager.load_data()

    info = dataset_manager.get_dataset_info()
    print("Dataset loaded successfully.")
    print(info)


if __name__ == "__main__":
    main()