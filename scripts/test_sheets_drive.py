import os
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH")
SHEET_ID = os.getenv("SHEET_ID")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


def get_credentials():
    return Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)


def test_sheets_connection(creds):
    print("\n--- Testing Google Sheets connection ---")
    try:
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.worksheet("Document Inventory")
        rows = worksheet.get_all_records()
        print(f"Connected successfully. Found {len(rows)} rows in 'Document Inventory'.")
        for row in rows[:3]:
            print(row)
    except Exception as e:
        print(f"Sheets connection failed: {e}")


def test_drive_connection(creds):
    print("\n--- Testing Google Drive connection ---")
    try:
        service = build("drive", "v3", credentials=creds)
        results = service.files().list(
            q=f"'{DRIVE_FOLDER_ID}' in parents",
            fields="files(id, name, mimeType)"
        ).execute()
        files = results.get("files", [])
        print(f"Connected successfully. Found {len(files)} items in the target folder.")
        for f in files[:10]:
            print(f"- {f['name']} ({f['mimeType']})")
    except Exception as e:
        print(f"Drive connection failed: {e}")


if __name__ == "__main__":
    creds = get_credentials()
    test_sheets_connection(creds)
    test_drive_connection(creds)