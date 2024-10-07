import unittest
from unittest.mock import patch, MagicMock
from app.attachment_processor import AttachmentProcessor
from app.exceptions import AttachmentProcessError
import io

class TestAttachmentProcessor(unittest.TestCase):
    @patch('PyPDF2.PdfReader')
    def test_process_pdf_attachment(self, mock_pdf_reader):
        # PDF 처리 모의 설정
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_pdf_reader.return_value.pages = [mock_page]

        attachment = {
            "content_type": "application/pdf",
            "content": b"fake pdf content"
        }

        result = AttachmentProcessor.process_attachment(attachment)
        self.assertEqual(result, "Test PDF content")

    @patch('openpyxl.load_workbook')
    def test_process_excel_attachment(self, mock_load_workbook):
        # Excel 처리 모의 설정
        mock_sheet = MagicMock()
        mock_sheet.iter_rows.return_value = [("Cell1", "Cell2")]
        mock_workbook = MagicMock()
        mock_workbook.sheetnames = ["Sheet1"]
        mock_workbook.__getitem__.return_value = mock_sheet
        mock_load_workbook.return_value = mock_workbook

        attachment = {
            "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "content": b"fake excel content"
        }

        result = AttachmentProcessor.process_attachment(attachment)
        self.assertEqual(result.strip(), "Cell1 Cell2")

    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    def test_process_image_attachment(self, mock_image_open, mock_image_to_string):
        # 이미지 처리 모의 설정
        mock_image_to_string.return_value = "Test image content"

        attachment = {
            "content_type": "image/jpeg",
            "content": b"fake image content"
        }

        result = AttachmentProcessor.process_attachment(attachment)
        self.assertEqual(result, "Test image content")

    def test_unsupported_attachment_type(self):
        attachment = {
            "content_type": "application/unknown",
            "content": b"unknown content"
        }

        result = AttachmentProcessor.process_attachment(attachment)
        self.assertEqual(result, "Unsupported file type")

    def test_attachment_process_error(self):
        attachment = {
            "content_type": "application/pdf",
            "content": b"invalid pdf content"
        }

        with patch('PyPDF2.PdfReader', side_effect=Exception("PDF processing error")):
            with self.assertRaises(AttachmentProcessError):
                AttachmentProcessor.process_attachment(attachment)

if __name__ == '__main__':
    unittest.main()