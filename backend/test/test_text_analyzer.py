import unittest
from unittest.mock import patch, MagicMock
from app.text_analyzer import TextAnalyzer
from app.exceptions import TextAnalysisError

class TestTextAnalyzer(unittest.TestCase):
    @patch('openai.ChatCompletion.create')
    def test_summarize_text_success(self, mock_openai):
        # OpenAI API 응답 모의 설정
        mock_response = MagicMock()
        mock_response.choices[0].message = {'content': 'Test summary'}
        mock_openai.return_value = mock_response

        text = "This is a long email that needs to be summarized."
        summary = TextAnalyzer.summarize_text(text)

        self.assertEqual(summary, 'Test summary')

    @patch('openai.ChatCompletion.create')
    def test_summarize_text_failure(self, mock_openai):
        # OpenAI API 오류 모의 설정
        mock_openai.side_effect = Exception("API error")

        text = "This is a long email that needs to be summarized."
        with self.assertRaises(TextAnalysisError):
            TextAnalyzer.summarize_text(text)

if __name__ == '__main__':
    unittest.main()