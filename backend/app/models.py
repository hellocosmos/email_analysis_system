from sqlalchemy import Column, Integer, String, Text, DateTime
from .database import Base

class Email(Base):
    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True)
    sender = Column(Text)
    recipient = Column(Text)
    cc = Column(Text)  # 새로 추가된 cc 필드
    subject = Column(String)
    body = Column(Text)
    summary = Column(Text)
    received_at = Column(DateTime)

class Attachment(Base):
    __tablename__ = "attachments"

    id = Column(Integer, primary_key=True, index=True)
    email_id = Column(Integer)
    filename = Column(String)
    content_type = Column(String)
    content = Column(Text)