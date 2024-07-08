import json
import re
import generate_html
import generate_html2
from send_mail import gmail_authenticate, send_message
import pathlib
import os

def main():
    service = gmail_authenticate()

    with open('improvement_msg.json', 'r') as file:
        improvement = json.load(file)

    for item in improvement:

        mail = item['email']
        msg = item['improvement_msg']
        name = item['name']

        data = {
            'name': name,
            'msg': msg
        }

        html = generate_html.generate_html(data)
        file_name = re.sub(r'\W+', '', mail) + '.html'

        with open(file_name, 'w', encoding="utf8") as file:
            file.write(html)

        send_message(service, mail, 'Your Application for Data Scientist position', file_name, image_path='images')


    with open('selected.json', 'r') as file:
        selected = json.load(file)

    for item in selected:
            name = item['name']
            mail = item['email']
        
            data = {
                'name': name,
            }

            attachment_p = pathlib.Path('attachments') / f'{name.replace(" ", "")}.jpg'
            print(os.path.exists(attachment_p))
        
            html = generate_html2.generate_html(data)
            file_name = re.sub(r'\W+', '', mail) + '.html'
        
            with open(file_name, 'w', encoding="utf8") as file:
                file.write(html)
        
            send_message(service, mail, 'Invitation to Interview for Data Scientist position', file_name, image_path='images', attachment_path=attachment_p)

