import socket
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow, Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
import os.path

# Define the scopes and the credentials file
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'drive_key.json'

def authenticate_gdrive():
    creds = None
    # Load existing credentials if they exist
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', 0))
            port = sock.getsockname()[1]
            sock.close()

            # Start the flow
            flow = InstalledAppFlow.from_client_secrets_file(
                'drive_key.json',
                scopes=['https://www.googleapis.com/auth/drive'],
                redirect_uri=f'http://localhost:{port}/')

            flow.run_local_server(port=port)
            creds = flow.credentials

            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def upload_file(file_name, file_path, creds):
    service = build('drive', 'v3', credentials=creds)
    file_metadata = {'name': file_name}
    media = MediaFileUpload(file_path, mimetype='video/mp4')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"File ID: {file.get('id')}")

# Authenticate and upload a file
creds = authenticate_gdrive()
upload_file('compressed_video.mp4', 'compressed_video.mp4', creds)
