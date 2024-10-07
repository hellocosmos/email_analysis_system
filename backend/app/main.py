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