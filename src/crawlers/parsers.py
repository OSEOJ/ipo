"""
텍스트 파싱 유틸리티 모듈
"""
import re
from typing import Optional
from datetime import datetime


def parse_shares_from_text(text: str) -> Optional[int]:
    """
    텍스트에서 주 수 파싱 (예: "1,234,567 주" -> 1234567)
    범위 형식인 경우 평균값 반환 (예: "800,000~1,000,000 주" -> 900000)

    Args:
        text: 파싱할 텍스트

    Returns:
        주 수 (int) 또는 None
    """
    if not text:
        return None

    # 범위 형식: "800,000~1,000,000 주" 또는 "800,000 ~ 1,000,000 주"
    range_match = re.search(r'([\d,]+)\s*[~\-]\s*([\d,]+)\s*주', text)
    if range_match:
        try:
            low = int(range_match.group(1).replace(',', ''))
            high = int(range_match.group(2).replace(',', ''))
            return (low + high) // 2
        except ValueError:
            pass

    # 단일 값: "1,234,567 주"
    single_match = re.search(r'([\d,]+)\s*주', text)
    if single_match:
        try:
            return int(single_match.group(1).replace(',', ''))
        except ValueError:
            pass

    return None


def parse_rate_from_text(text: str) -> Optional[float]:
    """
    텍스트에서 경쟁률 파싱 (예: "1,234.56:1" -> 1234.56)

    Args:
        text: 파싱할 텍스트

    Returns:
        경쟁률 (float) 또는 None
    """
    if not text:
        return None

    # "1,234.56:1" 또는 "1234.56 : 1" 형식
    match = re.search(r'([\d,\.]+)\s*:\s*1', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass

    # 숫자만 있는 경우 (예: "1,234.56")
    match = re.search(r'^([\d,\.]+)$', text.replace(',', '').strip())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def parse_percentage_from_text(text: str) -> Optional[float]:
    """
    텍스트에서 퍼센트 파싱 (예: "12.34%" -> 12.34)

    Args:
        text: 파싱할 텍스트

    Returns:
        퍼센트 값 (float) 또는 None
    """
    if not text:
        return None

    match = re.search(r'([\d\.]+)\s*%', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def parse_listing_date(date_str: str) -> Optional[datetime]:
    """다양한 형식의 날짜 문자열 파싱"""
    if not date_str or date_str == '-':
        return None
    
    # 다양한 날짜 형식 시도
    formats = [
        '%Y-%m-%d',      # 2025-12-29
        '%m/%d/%Y',      # 12/29/2025
        '%Y/%m/%d',      # 2025/12/29
        '%Y.%m.%d',      # 2025.12.29
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    return None


def parse_date(date_str: str) -> Optional[datetime]:
    """날짜 문자열을 datetime 객체로 변환 (기본 포맷)"""
    return parse_listing_date(date_str)
