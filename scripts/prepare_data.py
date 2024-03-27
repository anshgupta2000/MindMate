import pandas as pd
import requests
import json

def load_csv_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def fetch_json_data(url):
    """
    Fetch data from a JSON URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
        return None

def prepare_data_from_json(json_data):
    """
    Prepare data from JSON format.
    """
    rows = []
    for intent in json_data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            for response in intent['responses']:
                row = {'tag': tag, 'pattern': pattern, 'response': response}
                rows.append(row)
    return pd.DataFrame(rows)

def main():
    # Load CSV data from files
    patient_therapist_convs_df = load_csv_data('/Users/anshgupta/Desktop/MindMate/data/patient_therapist_convs.csv')
    synthetic_therapy_convs_df = load_csv_data('/Users/anshgupta/Desktop/MindMate/data/synthetic_therapy_convs.csv')

    # Fetch and prepare JSON data
    intents_url = "https://storage.googleapis.com/kagglesdsdata/datasets/2594075/4429121/intents.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240324%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240324T233046Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7c815915e7696c216c23ddd64f8186d299e1e7b5dc14ef7f4467dfedcbee9d6f0c3137c0c07dc2191487dad233a93ff30df52febe3bb0b6d170b9c521df79ab4075ec59adba33b967f2ab4f3d6d7660987cc250ee50ece0ba1d460b2c0793436dc0e7809b76fbae8e8cc7ef50aa2447456451bdfd655639f2c0fbf741975d44570cef66030ab29090b431263662c2e6d752da4279d4da3e5bfe11a3b3a1f8ff209bd01dbcb22775e2b2be420330ee691b5f5debd58179432c8da6fab55b7e16b528566edcf2259726f3a0a7f670bf4be42fcc4a120623af129e46b3e415e467a5f2e8a4fed7d269d9ba4535cc76c11b8526c4939ae4a32d2b0771600177a1551"
    json_data = fetch_json_data(intents_url)
    intents_df = prepare_data_from_json(json_data)

    print(patient_therapist_convs_df.head())
    print(synthetic_therapy_convs_df.head())
    print(intents_df.head())

if __name__ == "__main__":
    main()
