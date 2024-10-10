
import pandas as pd

# Load the dataset (replace 'path_to_your_drugbank_data.csv' with your actual file path)
def load_drug_disease_data(file_path='path_to_your_drugbank_data.csv'):
    drug_disease_df = pd.read_csv(file_path)
    return drug_disease_df

# Example usage
if __name__ == "__main__":
    data = load_drug_disease_data()
    print(data.head())
