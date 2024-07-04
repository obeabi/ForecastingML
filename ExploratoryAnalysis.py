import pandas as pd
#import pdfkit
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from Logger import CustomLogger


log_3 = CustomLogger()


class extensive_eda:
    """
    Custom class for extensive EDA

    """

    def __init__(self):
        self._data = None

    def perform_eda(self, data):

        """
        param:
        data
        Generate report as a widget
        """
        try:
            pf = ProfileReport(data)
            return pf.to_widgets()
        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            log_3.log("Something went wrong while saving the EDA performed on the dataset as a widget", level='ERROR')

    def save_eda_html(self, data):
        """
        param:
        data
         Save the report as an HTML file
        """
        try:
            profile = ProfileReport(data, title="EDA Report", explorative=True, dark_mode=True)
            return profile.to_file("EDA_Report.html")
        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            log_3.log("Something went wrong while saving the EDA performed on the dataset as a html file", level='ERROR')

    def __str__(self):
        return "This is my custom EDA class object"


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('Advertising.csv')
    df = df.drop(df.columns[0], axis=1)
    print(df)
    # Perform EDA
    #eda = extensive_eda()
    #eda.save_eda_html(df)
    # Print or save the report
    #print(eda)
