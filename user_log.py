import os
import pandas as pd
from datetime import datetime

log_directory = "user_logs"
excel_file = os.path.join(log_directory, "user_interactions.xlsx")

def log_user_interaction(user, action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "user": user,
        "action": action
    }

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    save_to_excel(log_entry)
    print(f"Logged interaction: {user} - {action}")

def save_to_excel(log_entry):
    try:
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
        else:
            df = pd.DataFrame(columns=["timestamp", "user", "action"])

        # Convert log_entry to DataFrame
        new_entry_df = pd.DataFrame([log_entry])

        # Concatenate the new entry with the existing DataFrame
        df = pd.concat([df, new_entry_df], ignore_index=True)

        # Save the DataFrame to an Excel file
        df.to_excel(excel_file, index=False)
    except Exception as e:
        print(f"An error occurred while saving to Excel: {e}")


