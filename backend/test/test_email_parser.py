import unittest
from app.email_parser import EmailParser
from app.exceptions import EmailParseError
from email.message import EmailMessage

class TestEmailParser(unittest.TestCase):
    def test_parse_email_success(self):
        # 테스트용 이메일 메시지 생성
        email_msg = EmailMessage()
        email_msg['Subject'] = 'Test Subject'
        email_msg['From'] = 'sender@example.com'
        email_msg['To'] = 'recipient@example.com'
        email_msg.set_content('Test body')

        parsed_email = EmailParser.parse_email(email_msg)

        self.assertEqual(parsed_email['subject'], 'Test Subject')
        self.assertEqual(parsed_email['sender'], 'sender@example.com')
        self.assertEqual(parsed_email['recipient'], 'recipient@example.com')
        self.assertEqual(parsed_email['body'], 'Test body')
        self.assertEqual(parsed_email['attachments'], [])

    def test_parse_email_with_attachment(self):
        # 첨부 파일이 있는 테스트용 이메일 메시지 생성
        email_msg = EmailMessage()
        email_msg['Subject'] = 'Test Subject with Attachment'
        email_msg['From'] = 'sender@example.com'
        email_msg['To'] = 'recipient@example.com'
        email_msg.set_content('Test body')
        
        # 첨부 파일 추가
        email_msg.add_attachment('Test attachment content'.encode(),
                                 filename='test.txt',
                                 maintype='text',
                                 subtype='plain')

        parsed_email = EmailParser.parse_email(email_msg)

        self.assertEqual(len(parsed_email['attachments']), 1)
        self.assertEqual(parsed_email['attachments'][0]['filename'], 'test.txt')

    def test_parse_email_failure(self):
        # 잘못된 형식의 이메일로 테스트
        invalid_email = "This is not a valid email message"

        with self.assertRaises(EmailParseError):
            EmailParser.parse_email(invalid_email)

if __name__ == '__main__':
    unittest.main()