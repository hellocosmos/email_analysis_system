import imaplib
import email
from email.header import decode_header
import os
from dotenv import load_dotenv
from .exceptions import EmailFetchError
from .logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

class EmailFetcher:
    def __init__(self):
        self.imap_server = os.getenv("IMAP_SERVER")
        self.email_address = os.getenv("EMAIL_ADDRESS")
        self.email_password = os.getenv("EMAIL_PASSWORD")

    def fetch_emails(self, num_emails=10):
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.email_address, self.email_password)
            mail.select("inbox")

            # 최신 이메일부터 가져오기 위해 역순으로 정렬
            _, search_data = mail.sort('REVERSE DATE', 'UTF-8', 'ALL')
            email_ids = search_data[0].split()

            # 지정된 수(num_emails)만큼 최신 이메일 ID 선택
            email_ids = email_ids[:num_emails]

            emails = []
            for email_id in email_ids:
                _, msg_data = mail.fetch(email_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        email_body = response_part[1]
                        email_message = email.message_from_bytes(email_body)
                        emails.append(email_message)

            mail.close()
            mail.logout()

            logger.info(f"Successfully fetched {len(emails)} emails")
            return emails
        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            raise EmailFetchError(f"Failed to fetch emails: {str(e)}")