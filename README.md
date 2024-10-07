# Simple Email Analysis & Viewer with ChatGPT

- The implementation of this code is freely available for personal use. If you wish to use it commercially, consultation with the author is required. When distributing, you must specify the author and license.
- Author : hellocosmos@gmail.com

## Summary 
This is a python code that analyzes emails to extract summaries and information from them.

See the attached MD file for code and explanation. 

- Currently, it only supports IMAP. 
- It uses GPT for summarizing email body. 
- Supports PDF, Excel files as attachments.

## Project Trees

```
email_analysis_system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── database.py
│   │   ├── email_fetcher.py
│   │   ├── email_parser.py
│   │   ├── text_analyzer.py
│   │   ├── attachment_processor.py
│   │   ├── exceptions.py
│   │   └── logger.py
│   └── logging.conf
├── frontend/
│   ├── app.py
└── .env

```

## Back-end Codes
### 0 .env
```
DATABASE_URL=postgresql://localhost/email_analysis_system
IMAP_SERVER=
EMAIL_ADDRESS=
EMAIL_PASSWORD=
OPENAI_API_KEY=
```

### 1. exceptions.py

```python
class EmailAnalysisException(Exception):
    """Base exception for email analysis system"""

class EmailFetchError(EmailAnalysisException):
    """Raised when there's an error fetching emails"""

class EmailParseError(EmailAnalysisException):
    """Raised when there's an error parsing an email"""

class TextAnalysisError(EmailAnalysisException):
    """Raised when there's an error analyzing text"""

class AttachmentProcessError(EmailAnalysisException):
    """Raised when there's an error processing an attachment"""
```

### 2. logger.py

```python
import logging
from logging.config import fileConfig
import os

def setup_logger(name):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'logging.conf')
    if os.path.exists(config_path):
        fileConfig(config_path)
    else:
        # 기본 로깅 설정
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)
```

### 3. email_fetcher.py

```python
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
```

### 4. email_parser.py

```python
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
```

### 5. text_analyzer.py

```python
from openai import OpenAI
from dotenv import load_dotenv
import os
from .exceptions import TextAnalysisError
from .logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

class TextAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def summarize_text(self, text):
        if not text or len(text.strip()) == 0:
            logger.warning("Attempted to summarize empty text")
            return "본문이 비어있습니다."

        try:
            prompt = f"""You are a helpful assistant that summarizes emails. Please summarize the following email content in Korean. Your summary should be concise yet comprehensive, capturing the main points of the email.

            Guidelines for summarization:
            1. Start with a brief statement of the email's main purpose or topic.
            2. Include key information such as important facts, dates, or numbers mentioned in the email.
            3. If there are any requested actions or next steps, clearly state them.
            4. Keep the overall summary within 25% of the original email's length.
            5. If technical terms or abbreviations are used, provide brief explanations.
            6. Maintain an objective and neutral tone in your summary.
            7. Use clear and professional language, regardless of the original email's style.

            Please provide your summary in Korean, following these guidelines.

            Email content to summarize:

            {text}

            Summary:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 또는 사용 가능한 최신 모델
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes emails. ANSWER IN KOREAN"},
                    {"role": "user", "content": prompt}
                ],
                # max_tokens=150,  # 요약의 최대 길이를 조절할 수 있습니다
                # n=1,
                # stop=None,
                temperature=0,
            )

            summary = response.choices[0].message.content.strip()
            logger.info("Successfully summarized text")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            raise TextAnalysisError(f"Failed to summarize text: {str(e)}")
```

### 6. attachment_processor.py

```python
import base64
import fitz  # PyMuPDF
from openpyxl import load_workbook
import pytesseract
from PIL import Image
import io
from .exceptions import AttachmentProcessError
from .logger import setup_logger

logger = setup_logger(__name__)

class AttachmentProcessor:
    @staticmethod
    def process_attachment(attachment):
        try:
            content_type = attachment["content_type"]
            content = base64.b64decode(attachment["content"])

            if content_type == "application/pdf":
                return AttachmentProcessor._process_pdf(content)
            elif content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                return AttachmentProcessor._process_excel(content)
            elif content_type.startswith("image/"):
                return AttachmentProcessor._process_image(content)
            elif content_type == "text/plain":
                return AttachmentProcessor._process_text(content)
            else:
                logger.warning(f"Unsupported file type: {content_type}")
                return f"Unsupported file type: {content_type}"
        except Exception as e:
            logger.error(f"Error processing attachment: {str(e)}")
            raise AttachmentProcessError(f"Failed to process attachment: {str(e)}")

    @staticmethod
    def _process_pdf(content):
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text
                else:  # If no text found, try OCR
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img, lang='kor+eng')  # Added Korean language support

            logger.info("Successfully processed PDF attachment")
            return AttachmentProcessor._clean_text(text)
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return "PDF 처리 중 오류가 발생했습니다."

    @staticmethod
    def _process_excel(content):
        workbook = load_workbook(filename=io.BytesIO(content))
        text = ""
        for sheet in workbook.sheetnames:
            active_sheet = workbook[sheet]
            for row in active_sheet.iter_rows(values_only=True):
                text += " ".join(str(cell) for cell in row if cell is not None) + "\n"
        logger.info("Successfully processed Excel attachment")
        return AttachmentProcessor._clean_text(text)

    @staticmethod
    def _process_image(content):
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)
        logger.info("Successfully processed image attachment")
        return AttachmentProcessor._clean_text(text)

    @staticmethod
    def _process_text(content):
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            encodings = ['ascii', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error("Failed to decode text file with all attempted encodings")
                return "Unable to decode text file"
        
        logger.info("Successfully processed text attachment")
        return AttachmentProcessor._clean_text(text)

    @staticmethod
    def _clean_text(text):
        cleaned_text = text.replace('\x00', '')
        cleaned_text = ''.join(char for char in cleaned_text if ord(char) >= 32 or char == '\n')
        return cleaned_text```

### 7. main.py

```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, schemas
from .database import engine, get_db
from .email_fetcher import EmailFetcher
from .email_parser import EmailParser
from .text_analyzer import TextAnalyzer
from .attachment_processor import AttachmentProcessor
from .exceptions import EmailAnalysisException
from .logger import setup_logger
from pydantic import BaseModel
from datetime import datetime
import json
import base64

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
logger = setup_logger(__name__)

text_analyzer = TextAnalyzer()

class EmailServerSettings(BaseModel):
    imap_server: str
    email_address: str
    email_password: str

class FetchEmailsRequest(BaseModel):
    num_emails: int

def decode_base64(encoded_text):
    try:
        return base64.b64decode(encoded_text).decode('utf-8')
    except:
        return encoded_text

@app.post("/update_email_settings")
def update_email_settings(settings: EmailServerSettings):
    try:
        import os
        os.environ["IMAP_SERVER"] = settings.imap_server
        os.environ["EMAIL_ADDRESS"] = settings.email_address
        os.environ["EMAIL_PASSWORD"] = settings.email_password
        
        logger.info("Email server settings updated successfully")
        return {"message": "Email server settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating email server settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch_and_process_emails")
def fetch_and_process_emails(request: FetchEmailsRequest, db: Session = Depends(get_db)):
    try:
        email_fetcher = EmailFetcher()
        emails = email_fetcher.fetch_emails(request.num_emails)

        for email_message in emails:
            try:
                parsed_email = EmailParser.parse_email(email_message)
                logger.info(f"Parsed email: {parsed_email['subject']}")
                logger.info(f"Body length: {len(parsed_email['body'])}")
                logger.info(f"Attachments: {[att['filename'] for att in parsed_email['attachments']]}")

                
                # 이메일 본문 디코딩
                parsed_email["body"] = EmailParser.decode_body(parsed_email["body"])
                
                summary = text_analyzer.summarize_text(parsed_email["body"])

                received_at = parsed_email["received_at"]
                if received_at and isinstance(received_at, datetime):
                    received_at = received_at.isoformat()
                else:
                    received_at = datetime.now().isoformat()

                email_model = models.Email(
                    sender=json.dumps(parsed_email["sender"]),
                    recipient=json.dumps(parsed_email["recipient"]),
                    cc=json.dumps(parsed_email.get("cc", [])),
                    subject=parsed_email["subject"],
                    body=parsed_email["body"],
                    summary=summary,
                    received_at=received_at
                )
                db.add(email_model)
                db.commit()

                for attachment in parsed_email["attachments"]:
                    logger.info(f"Processing attachment: {attachment['filename']} for email ID: {email_model.id}")
                    try:
                        processed_content = AttachmentProcessor.process_attachment(attachment)
                        attachment_model = models.Attachment(
                            email_id=email_model.id,
                            filename=attachment["filename"],
                            content_type=attachment["content_type"],
                            content=processed_content
                        )
                        db.add(attachment_model)
                        logger.info(f"Successfully added attachment: {attachment['filename']} for email ID: {email_model.id}")
                    except Exception as e:
                        logger.error(f"Error processing attachment {attachment['filename']} for email ID {email_model.id}: {str(e)}")

                if not parsed_email["attachments"]:
                    logger.info(f"No attachments found for email ID: {email_model.id}")

                db.commit()
                logger.info(f"Committed attachments for email ID: {email_model.id}")
            except Exception as e:
                logger.error(f"Error processing email: {str(e)}")
                continue

        logger.info(f"Fetched and processed {len(emails)} emails successfully")
        return {"message": f"Fetched and processed {len(emails)} emails successfully"}
    except EmailAnalysisException as e:
        logger.error(f"Error in fetch_and_process_emails: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in fetch_and_process_emails: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.get("/emails")
def get_emails(db: Session = Depends(get_db)):
    try:
        emails = db.query(models.Email).all()
        logger.info("Successfully retrieved all emails")
        return emails
    except Exception as e:
        logger.error(f"Error retrieving emails: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/emails/{email_id}")
def get_email(email_id: int, db: Session = Depends(get_db)):
    try:
        email = db.query(models.Email).filter(models.Email.id == email_id).first()
        if email is None:
            logger.warning(f"Email with id {email_id} not found")
            raise HTTPException(status_code=404, detail="Email not found")
        
        # 이메일 본문 디코딩
        email.body = decode_base64(email.body)
        
        logger.info(f"Successfully retrieved email with id {email_id}")
        return email
    except Exception as e:
        logger.error(f"Error retrieving email with id {email_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/emails/{email_id}/attachments")
def get_email_attachments(email_id: int, db: Session = Depends(get_db)):
    try:
        attachments = db.query(models.Attachment).filter(models.Attachment.email_id == email_id).all()
        
        # 첨부 파일 내용 디코딩
        for attachment in attachments:
            attachment.content = decode_base64(attachment.content)
        
        logger.info(f"Successfully retrieved attachments for email with id {email_id}")
        return attachments
    except Exception as e:
        logger.error(f"Error retrieving attachments for email with id {email_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 8. logging.conf

```ini
[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_emailAnalysisSystem]
level=DEBUG
handlers=consoleHandler,fileHandler
qualname=emailAnalysisSystem
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=('email_analysis_system.log', 'a')

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=
```

## Front-end Code (Streamlit)

### app.py

```python
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
import base64
import html

BASE_URL = "http://localhost:8000"  # FastAPI 백엔드 URL

st.set_page_config(page_title="이메일 분석 시스템", page_icon="📧", layout="wide")

st.title("이메일 분석 시스템")

menu = st.sidebar.radio(
    "메뉴 선택",
    ("이메일 가져오기", "이메일 목록", "이메일 상세 정보")
)

def safe_json_load(x):
    if not x:
        return {'name': '', 'email': ''}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {x}")
        return {'name': x, 'email': ''}

def decode_base64(encoded_text):
    try:
        return base64.b64decode(encoded_text).decode('utf-8')
    except:
        return encoded_text

if menu == "이메일 가져오기":
    st.header("이메일 가져오기 및 처리")
    num_emails = st.number_input("가져올 이메일 수", min_value=1, max_value=50, value=3, step=1)
    if st.button("이메일 가져오기 시작"):
        with st.spinner("이메일을 가져오는 중..."):
            response = requests.post(f"{BASE_URL}/fetch_and_process_emails", json={"num_emails": num_emails})
            if response.status_code == 200:
                st.success(f"{response.json()['message']}")
            else:
                st.error(f"이메일 처리 중 오류가 발생했습니다: {response.text}")

elif menu == "이메일 목록":
    st.header("처리된 이메일 목록")
    emails_response = requests.get(f"{BASE_URL}/emails")
    if emails_response.status_code == 200:
        emails = emails_response.json()
        if emails:
            df = pd.DataFrame(emails)
            df['received_at'] = pd.to_datetime(df['received_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['sender'] = df['sender'].apply(lambda x: safe_json_load(x)['name'])
            df['recipient'] = df['recipient'].apply(lambda x: safe_json_load(x)['name'])
            st.dataframe(df[["id", "sender", "recipient", "subject", "received_at"]])
        else:
            st.info("처리된 이메일이 없습니다.")
    else:
        st.error("이메일 목록을 가져오는 중 오류가 발생했습니다.")

elif menu == "이메일 상세 정보":
    st.header("이메일 상세 정보")
    email_id = st.number_input("이메일 ID를 입력하세요", min_value=1, step=1)
    if st.button("이메일 정보 가져오기"):
        email_response = requests.get(f"{BASE_URL}/emails/{email_id}")
        if email_response.status_code == 200:
            email = email_response.json()
            st.subheader(f"제목: {email['subject']}")
            sender = safe_json_load(email['sender'])
            recipient = safe_json_load(email['recipient'])
            cc = safe_json_load(email['cc']) if email['cc'] else []
            st.write(f"보낸 사람: {sender['name']} ({sender['email']})")
            st.write(f"받는 사람: {recipient['name']} ({recipient['email']})")
            if cc:
                st.write("CC:")
                for person in cc:
                    st.write(f"- {person['name']} ({person['email']})")
            st.write(f"수신 시간: {datetime.fromisoformat(email['received_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            with st.expander("이메일 본문", expanded=True):
                if email['body'] and email['body'].strip():
                    # HTML 이스케이프 처리를 추가하여 안전하게 표시
                    safe_body = html.escape(email['body'])
                    st.markdown(safe_body, unsafe_allow_html=True)
                else:
                    st.info("본문이 비어있습니다.")
            
            with st.expander("이메일 요약", expanded=True):
                if email['summary'] and email['summary'].strip():
                    st.text_area("", value=email['summary'], height=150)
                else:
                    st.info("요약을 생성할 수 없습니다.")

            attachments_response = requests.get(f"{BASE_URL}/emails/{email_id}/attachments")
            if attachments_response.status_code == 200:
                attachments = attachments_response.json()
                if attachments:
                    st.subheader("첨부 파일")
                    for attachment in attachments:
                        with st.expander(f"첨부 파일: {attachment['filename']}", expanded=True):
                            st.write(f"파일 유형: {attachment['content_type']}")
                            if attachment['content']:
                                st.text_area("", value=decode_base64(attachment['content']), height=200)
                            else:
                                st.warning("첨부 파일의 내용을 표시할 수 없습니다.")
                else:
                    st.info("첨부 파일이 없습니다.")
            else:
                st.error(f"첨부 파일 정보를 가져오는 중 오류가 발생했습니다. 상태 코드: {attachments_response.status_code}")
        elif email_response.status_code == 404:
            st.warning("해당 ID의 이메일을 찾을 수 없습니다.")
        else:
            st.error("이메일 정보를 가져오는 중 오류가 발생했습니다.")

st.sidebar.markdown("---")
st.sidebar.info(
    "이 시스템은 이메일을 자동으로 가져와 분석하고, "
    "첨부 파일을 처리하여 중요한 정보를 추출합니다. "
    "GPT-4o-mini를 이용한 이메일 요약 기능도 포함되어 있습니다."
)
st.sidebar.text("버전: 1.1.0")
st.sidebar.text(f"현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

Step by Setp:
 
1. PIP List
   ```
    langchain                                0.3.0
    langchain-anthropic                      0.2.1
    langchain-community                      0.3.0
    langchain-core                           0.3.5
    langchain-openai                         0.2.0
    langchain-text-splitters                 0.3.0
    langdetect                               1.0.9
    langsmith                                0.1.126
    streamlit                                1.39.0 
    ```

2. DB Installation : 
   ```
    - PostgreSQL 설치:
    맥OS에서 Homebrew를 사용하여 PostgreSQL을 설치할 수 있습니다. 최신 버전(현재 PostgreSQL 15)을 설치하려면 다음 명령어를 사용하세요:
    brew install postgresql
    
    - PostgreSQL 서비스 시작:
    설치 후 PostgreSQL 서비스를 시작합니다:
    brew services start postgresql
    
    - PostgreSQL 서비스 중지:
    brew services stop postgresql
   ```
   
3. Execute Back-end:
   ```
   cd backend
   uvicorn app.main:app --reload
   ```

4. Execute Front-end:
   ```
   cd frontend
   streamlit run app.py
   ```
