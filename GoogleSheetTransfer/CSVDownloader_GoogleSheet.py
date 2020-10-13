# Revise the reference google sheet API v4(Python Quickstart): https://developers.google.com/sheets/api/quickstart/python?authuser=1
from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import csv

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '1v7wgag0KPQbVAY1z6RB-5sWkkydnAKyE4OtD_7Ue-Ug'
RANGE_NAME_1 = 'Newest datas!A1:C'
RANGE_NAME_2 = 'History Humidity!A1:B'
RANGE_NAME_3 = 'History Temperature!A1:B'

def main():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '../../_Credentials/google_sheet_credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    ## Newest datas
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=RANGE_NAME_1).execute()
    values = result.get('values', [])
    ## History Humidity
    result2 = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=RANGE_NAME_2).execute()
    values2 = result2.get('values', [])
    ## History Temperature
    result3 = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=RANGE_NAME_3).execute()
    values3 = result3.get('values', [])

    if not values:
        print('No data found.')
    else:
        # Print columns A and E, which correspond to indices 0 and 4.
        # print("{}{}".format(row[0], row[1]))
        with open('DHT11_NewestData.csv', 'w', encoding='utf8',newline='') as gsfile:
            gswriter = csv.writer(gsfile, dialect='excel')
            for row in values:
                gswriter.writerow(['{}'.format(row[0]),
                                   '{}'.format(row[1]),
                                   '{}'.format(row[2])
                                  ])
        with open('DHT11_HistoryHumidity.csv', 'w', encoding='utf8',newline='') as gsfile:
            gswriter = csv.writer(gsfile, dialect='excel')
            for row in values2:
                gswriter.writerow(['{}'.format(row[0]),
                                   '{}'.format(row[1])
                                  ])
        with open('DHT11_HistoryTemperatrue.csv', 'w', encoding='utf8',newline='') as gsfile:
            gswriter = csv.writer(gsfile, dialect='excel')
            for row in values3:
                gswriter.writerow(['{}'.format(row[0]),
                                   '{}'.format(row[1])
                                  ])                                
        print('Downloaded and write into csv successfully!')
            
if __name__ == '__main__':
    main()
