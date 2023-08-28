from flask import Flask
from flask_mail import Mail, Message
import os
from pathlib import Path

app = Flask(__name__)
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": os.environ['GMAIL_USER'],
    "MAIL_PASSWORD": os.environ['GMAIL_PASSWORD']
}
app.config.update(mail_settings)
mail = Mail(app)

def send_mail(alerts, file_attachment1, file_attachment2, file_attachment3, file_attachment4,
              file_attachment5, file_attachment6):
    with app.app_context():
        msg = Message(subject="Forecast and Current Snowpack",
                      sender=app.config.get("MAIL_USERNAME"),
                      reply_to="smotley@pcwa.net",
                      bcc=["kswanberg@pcwa.net",
                            "bbarker@pcwa.net",
                           "afecko@pcwa.net",
                           "dreintjes@pcwa.net",
                           # "bransom@pcwa.net",
                           # "asullivan@pcwa.net",
                           # "kdushane@pcwa.net",
                           # "pcannarozzi@pcwa.net",
                           # "jbergman@pcwa.net"
                           ], # replace with your email for testing
                      recipients=["smotley@pcwa.net"], # replace with your email for testing
                      body="Max Temps and Precip Forecast for Sacramento are attached.")
        msg.html = alerts

        # Sacramento Temp Image
        msg.attach(file_attachment1, "image/png", app.open_resource(file_attachment1).read())

        # Burbank Temp Image
        # msg.attach(file_attachment2, "image/png", app.open_resource(file_attachment2).read())
        if Path(file_attachment2).exists():
            msg.attach(file_attachment2, "image/png", app.open_resource(file_attachment2).read())
        if Path(file_attachment3).exists():
            msg.attach(file_attachment3, "image/png", app.open_resource(file_attachment3).read())
        if Path(file_attachment4).exists():
            msg.attach(file_attachment4, "image/jpg", app.open_resource(file_attachment4).read())
        if Path(file_attachment5).exists():
            msg.attach(file_attachment5, "image/jpg", app.open_resource(file_attachment5).read())
        if Path(file_attachment6).exists():
            msg.attach(file_attachment6, "image/png", app.open_resource(file_attachment6).read())
        mail.send(msg)
        print("MAIL SENT")
    return

if __name__ == '__main__':
    send_mail()

