import base64
import quopri
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from .exceptions import EmailParseError
from .logger import setup_logger
import chardet

logger = setup_logger(__name__)

class EmailParser:
    @staticmethod
    def decode_str(s):
        if s is None:
            return ""
        try:
            decoded_list = decode_header(s)
            decoded_parts = []
            for content, charset in decoded_list:
                if isinstance(content, bytes):
                    if charset is None or charset == 'unknown-8bit':
                        detected = chardet.detect(content)
                        charset = detected['encoding']
                    if charset is None:
                        charset = 'utf-8'
                    decoded_parts.append(content.decode(charset, errors='replace'))
                else:
                    decoded_parts.append(str(content))
            return ''.join(decoded_parts)
        except Exception as e:
            logger.warning(f"Error decoding string: {e}")
            return str(s)

    @staticmethod
    def parse_email_address(address):
        name, email = parseaddr(address)
        return {
            "name": EmailParser.decode_str(name),
            "email": email
        }

    @staticmethod
    def decode_base64(encoded_text):
        try:
            # Base64 디코딩 시도
            decoded = base64.b64decode(encoded_text).decode('utf-8')
            return decoded
        except:
            # 디코딩 실패 시 원본 반환
            return encoded_text

    @staticmethod
    def decode_body(body):
        if body is None:
            return ""
        
        # base64 디코딩 시도
        try:
            return base64.b64decode(body).decode('utf-8')
        except:
            pass

        # quoted-printable 디코딩 시도
        try:
            return quopri.decodestring(body).decode('utf-8')
        except:
            pass

        # 그 외의 경우, 원본 반환
        return body
    
    @staticmethod
    def parse_email(email_message):
        try:
            subject = EmailParser.decode_str(email_message["Subject"])
            sender = EmailParser.parse_email_address(email_message["From"])
            recipient = EmailParser.parse_email_address(email_message["To"])
            cc = [EmailParser.parse_email_address(addr) for addr in email_message.get_all("Cc", [])]
            received_at = email_message["Date"]
            if received_at:
                received_at = parsedate_to_datetime(received_at)

            body = ""
            attachments = []

            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        part_body = part.get_payload(decode=True)
                        if isinstance(part_body, bytes):
                            part_body = part_body.decode('utf-8', errors='replace')
                        body += part_body + "\n"
                    elif "attachment" in content_disposition or part.get_filename():
                        filename = part.get_filename()
                        if filename:
                            attachments.append({
                                "filename": EmailParser.decode_str(filename),
                                "content_type": content_type,
                                "content": base64.b64encode(part.get_payload(decode=True)).decode()
                            })
                            logger.info(f"Found attachment: {filename}")
            else:
                body = email_message.get_payload(decode=True)

            if isinstance(body, bytes):
                body = body.decode('utf-8', errors='replace')

            # Base64 및 Quoted-Printable 디코딩 적용
            body = EmailParser.decode_body(body)

            logger.info(f"Successfully parsed email: {subject}")
            logger.info(f"Body length: {len(body)}")
            logger.info(f"Number of attachments: {len(attachments)}")

            return {
                "subject": subject,
                "sender": sender,
                "recipient": recipient,
                "cc": cc,
                "body": body,
                "attachments": attachments,
                "received_at": received_at
            }
        except Exception as e:
            logger.error(f"Error parsing email: {str(e)}")
            raise EmailParseError(f"Failed to parse email: {str(e)}")