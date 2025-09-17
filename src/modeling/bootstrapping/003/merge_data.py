import pandas as pd
from pathlib import Path

# MODEL Path
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")
KAGGLE_DATA = PROJECT_PATH / "data/interim/summarize_text/kaggle_data.csv"
SCRAPING_DATA1 = PROJECT_PATH / "data/interim/summarize_text/scraping-data-1.csv"
SCRAPING_DATA2 = PROJECT_PATH / "data/interim/summarize_text/scraping-data-2.csv"
SCRAPING_DATA3 = PROJECT_PATH / "data/interim/summarize_text/scraping-data-3.csv"
SCRAPING_DATA_EMBEDDED = (
    PROJECT_PATH / "data/interim/summarize_text/scraping-data-embedded.csv"
)
SCRAPING_DATA_IOT = PROJECT_PATH / "data/interim/summarize_text/scraping-data-iot.csv"
SCRAPING_DATA_ROBOT = (
    PROJECT_PATH / "data/interim/summarize_text/scraping-data-robot.csv"
)
SCRAPING_DATA_ARDUINO = (
    PROJECT_PATH / "data/interim/summarize_text/scraping-data-arduino.csv"
)


# Merge the dataframes and remove duplicates
def merge_dataframes():
    # Read the CSV files
    df1 = pd.read_csv(KAGGLE_DATA)
    df2 = pd.read_csv(SCRAPING_DATA1)
    df3 = pd.read_csv(SCRAPING_DATA2)
    df4 = pd.read_csv(SCRAPING_DATA3)
    df5 = pd.read_csv(SCRAPING_DATA_EMBEDDED)
    df6 = pd.read_csv(SCRAPING_DATA_IOT)
    df7 = pd.read_csv(SCRAPING_DATA_ROBOT)
    df8 = pd.read_csv(SCRAPING_DATA_ARDUINO)

    # Concatenate the dataframes
    merged_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    # Remove duplicates based on 'Qualification_Summary' column
    merged_df.drop_duplicates(subset=["Qualification_Summary"], inplace=True)

    # Save the merged dataframe to a new CSV file
    output_path = PROJECT_PATH / "data/interim/summarize_text/merged_data.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")


merge_dataframes()
