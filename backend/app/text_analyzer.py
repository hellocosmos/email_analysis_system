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