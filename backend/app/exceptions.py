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