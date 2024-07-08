import os
import mimetypes
import pickle
import glob
import pathlib

import lxml

from base64 import urlsafe_b64decode, urlsafe_b64encode

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from mimetypes import guess_type as guess_mime_type

from googleapiclient.discovery import build

our_email = 'rapidvortex219@gmail.com'


def gmail_authenticate():
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    return build('gmail', 'v1', credentials=creds)

# service = gmail_authenticate()


def add_attachment(message, filename):
    content_type, encoding = guess_mime_type(filename)
    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    main_type, sub_type = content_type.split('/', 1)
    if main_type == 'text':
        fp = open(filename, 'rb')
        msg = MIMEText(fp.read().decode(), _subtype=sub_type)
        fp.close()
    elif main_type == 'image':
        fp = open(filename, 'rb')
        msg = MIMEImage(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'audio':
        fp = open(filename, 'rb')
        msg = MIMEAudio(fp.read(), _subtype=sub_type)
        fp.close()
    else:
        fp = open(filename, 'rb')
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())
        fp.close()
    filename = os.path.basename(filename)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

def build_message(destination, obj, html_file_path, image_path=None, attachment_path=None):
    with open(html_file_path, 'r', encoding="utf8") as file:
        html_content = file.read()

    message = MIMEMultipart()
    message['to'] = destination
    message['from'] = our_email
    message['subject'] = obj
    message.attach(MIMEText(html_content, 'html'))

    if image_path is not None:
        image_path = pathlib.Path(image_path)
        image_paths = glob.glob(str(image_path / '*'))

        for img_path in image_paths:
            with open(img_path, 'rb') as img:
                img_data = img.read()
                img_name = os.path.basename(img_path)
                mime_type, _ = mimetypes.guess_type(img_path)
                main_type, sub_type = mime_type.split('/', 1)
                img_attachment = MIMEImage(img_data, _subtype=sub_type)
                img_attachment.add_header('Content-ID', f'<{img_name}>')
                img_attachment.add_header('Content-Disposition', 'inline', filename=img_name)
                message.attach(img_attachment)
    
    if attachment_path is not None:
        attachment_path = pathlib.Path(attachment_path)
        # attachments = glob.glob(str(attachment_path / '*'))
        if os.path.exists(attachment_path):
            add_attachment(message, attachment_path)
        # for filename in attachments:

    return {'raw': urlsafe_b64encode(message.as_bytes()).decode()}

def send_message(service, destination, obj, html_file_path, image_path=None, attachment_path=None):
    return service.users().messages().send(
      userId="me",
      body=build_message(destination, obj, html_file_path, image_path, attachment_path)
    ).execute()

# send_message(service, our_email, "This is a subject",
#             "generated_email.html")