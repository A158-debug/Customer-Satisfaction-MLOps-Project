from pipelines.training_pipeline import train_pipeline

# from zenml.client import Client
if __name__ == '__main__':
    # Run the pipeline
    # print(Client.activate_stack.experiment_tracker.get_tracking_uri())
    data_path = './data/olist_customers_dataset.csv'
    train_pipeline(data_path)