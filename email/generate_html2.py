from bs4 import BeautifulSoup
import json

def generate_html(data):

    with open('geminicode2.html', 'r', encoding="utf8") as file:
        html_content = file.read()

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the placeholder element and insert the generated text
    placeholder = soup.find(id='applicant-name')
    placeholder.string = f"Dear {data['name']},"

    return str(soup)
