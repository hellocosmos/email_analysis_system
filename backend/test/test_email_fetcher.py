import unittest
from unittest.mock import patch, MagicMock
from app.email_fetcher import EmailFetcher
from app.exceptions import EmailFetchError

class TestEmailFetcher(unittest.TestCase):
    @patch('imaplib.IMAP4_SSL')
    def test_fetch_emails_success(self, mock_imap):
        # 모의 이메일 메시지 설정
        mock_email = MagicMock()
        mock_email.__getitem__.return_value = "Test Subject"

        # IMAP 서버 응답 모의 설정
        mock_imap.return_value.search.return_value = ('OK', [b'1 2 3'])
        mock_imap.return_value.fetch.return_value = ('OK', [(b'1 (RFC822 {1234}', mock_email), b')'])

        fetcher = EmailFetcher()
        emails = fetcher.fetch_emails()

        self.assertEqual(len(emails), 3)
        self.assertEqual(emails[0]['Subject'], "Test Subject")

    @patch('imaplib.IMAP4_SSL')
    def test_fetch_emails_failure(self, mock_imap):
        mock_imap.return_value.login.side_effect = Exception("Login failed")

        fetcher = EmailFetcher()
        with self.assertRaises(EmailFetchError):
            fetcher.fetch_emails()

if __name__ == '__main__':
    unittest.main()