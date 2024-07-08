import json

data = [
    {
        "email": "example1@example.com",
        "message": "Hello, this is your message!"
    },
    {
        "email": "example2@example.com",
        "message": "Hi there, another message for you!"
    },
    {
        "email": "example3@example.com",
        "message": "Greetings, here's a different message!"
    }
]

with open('data.json', 'w') as file:
    json.dump(data, file, indent=4)