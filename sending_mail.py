import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
student_data = pd.read_csv('student_details.csv')
attendance_data = pd.read_csv('Attendance_summary.csv')
merged_data = pd.merge(student_data, attendance_data, on='Names')
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('keshavsunkara21@gmail.com', 'uzerdwtqugdzlvrx')
for index, row in merged_data.iterrows():
    name = row['Names']
    email = row['mail']
    attendance = row['Attendance']
    if attendance == 'Yes':
        message = MIMEMultipart()
        message['From'] = 'keshavsunkara21@gmail.com'
        message['To'] = email
        message['Subject'] = 'Attendance for Today'
        body = f"Hi {name},\n\nYou were present for today's session. Keep up the good work!\n\nRegards,\nYour Instructor"
        message.attach(MIMEText(body, 'plain'))
        text = message.as_string()
        server.sendmail('keshavsunkara21@gmail.com', email, text)
    else:
        message = MIMEMultipart()
        message['From'] = 'keshavsunkara21@gmail.com'
        message['To'] = email
        message['Subject'] = 'Attendance for Today'
        body = f"Hi {name},\n\nYou were absent for today's session. Please make sure to catch up with the missed material.\n\nRegards,\nYour Instructor"
        message.attach(MIMEText(body, 'plain'))
        text = message.as_string()
        server.sendmail('keshavsunkara21@gmail.com', email, text)

# Close the email server
server.quit()







