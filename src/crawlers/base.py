"""
크롤러 기본 유틸리티 모듈
"""
import subprocess
import requests
import urllib3
from typing import Optional

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_with_curl(url: str) -> Optional[str]:
    """
    curl을 사용하여 URL 내용 가져오기
    
    참고: www.38.co.kr 서버는 약한 DH 키를 사용하여 최신 OpenSSL에서 연결 거부됨.
    --ciphers DEFAULT@SECLEVEL=0 옵션으로 보안 레벨을 낮춰서 연결 허용.
    """
    try:
        result = subprocess.run(
            ['curl', '-s', '-k', '-L', '--ciphers', 'DEFAULT@SECLEVEL=0', url],
            capture_output=True,
            text=False,
            timeout=60
        )
        if result.returncode == 0 and result.stdout:
            # 한국어 인코딩 시도
            for encoding in ['utf-8', 'euc-kr', 'cp949']:
                try:
                    return result.stdout.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return result.stdout.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"[WARNING] curl 실행 에러: {e}")
    return None
