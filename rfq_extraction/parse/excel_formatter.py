# Importing the required libraries
import pandas as pd
from pandas.io.excel._xlsxwriter import XlsxWriter
import re

class RichExcelWriter(XlsxWriter):
    def __init__(self, *args, **kwargs):
        super(RichExcelWriter, self).__init__(*args, **kwargs)

    def _value_with_fmt(self, val):
        if type(val) == list:
            return val, None
        return super(RichExcelWriter, self)._value_with_fmt(val)

    def _write_cells(self, cells, sheet_name=None, startrow=0, startcol=0, freeze_panes=None):
        sheet_name = self._get_sheet_name(sheet_name)
        if sheet_name in self.sheets:
            wks = self.sheets[sheet_name]
        else:
            wks = self.book.add_worksheet(sheet_name)
            wks.set_column(0, 0, 40)
            wks.set_column(1, 5, 50)
            #add handler to the worksheet when it's created
            wks.add_write_handler(list, lambda worksheet, row, col, list, style: worksheet._write_rich_string(row, col, *list))
            self.sheets[sheet_name] = wks
        super(RichExcelWriter, self)._write_cells(cells, sheet_name, startrow, startcol, freeze_panes)


def create_excel_with_formatting(df, filename, sheet_name):
    """
    The create_excel_with_formatting function takes a DataFrame, filename, and sheet_name as input.
    It then creates an Excel file with the specified name and adds a worksheet to it with the specified name.
    The function then applies bold formatting to any text in the DataFrame that is surrounded by HTML <b></b> tags.
    
    :param df: Pass in the dataframe that will be converted to excel
    :param filename: Name the excel file that will be created
    :param sheet_name: Name the sheet in the excel file
    :return: A pandas excelwriter object
    """
    writer = RichExcelWriter(filename)
    workbook = writer.book
    bold = workbook.add_format({'bold': True})


    # Function to convert HTML bold tags to Excel bold formatting
    def convert_html_tags(text):
        """
        The convert_html_tags function takes a string as input and returns the same string with HTML tags converted to Excel formatting.
        
        :param text: Pass in the text that will be formatted
        :return: A list of formatted strings
        """
        if isinstance(text, float):
            return ' '
        if '<b>' not in text:
            return text
        parts = re.split(r'(<b>|</b>)', text)
        formatted_parts = [bold if part == '<b>' else part for part in parts if part != '</b>']
        return formatted_parts


    # Apply the function to each cell in the DataFrame
    for col in df.columns:
        df[col] = df[col].apply(convert_html_tags)

    output = df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.close()
    return output