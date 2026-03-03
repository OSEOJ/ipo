"""
DART 웹 크롤링 모듈
투자설명서 검색 및 PDF 다운로드

OpenDART API가 2020년 이전 투자설명서를 제공하지 않는 문제를 해결하기 위해
DART 웹사이트를 직접 크롤링하여 투자설명서 PDF를 다운로드합니다.

알고리즘:
1. 상장일 기준 년도에서 투자설명서 검색 (DART 웹 검색)
2. JavaScript에서 dcmNo 추출
3. PDF 직접 다운로드
"""
import re
import requests
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from bs4 import BeautifulSoup

from ..config import CACHE_DIR


class DartCrawler:
    """DART 웹 크롤링 클라이언트"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        })
        self.timeout = 30
        
        # PDF 다운로드 캐시 디렉토리
        self.pdf_cache_dir = CACHE_DIR / "pdf"
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _clean_company_name(self, name: str) -> str:
        """
        종목명 정제: (구.XXX), (유가) 등 불필요한 부분 제거
        예: "기가레인(구.맥시스)" -> "기가레인"
        """
        if not name:
            return name
        
        # (구.XXX), (舊XXX), (유가) 등 제거
        cleaned = re.sub(r'\(구\..*?\)', '', name)
        cleaned = re.sub(r'\(舊.*?\)', '', cleaned)
        cleaned = re.sub(r'\(유가\)', '', cleaned)
        
        return cleaned.strip()
    
    def search_prospectus(
        self,
        company_name: str,
        listing_date: str,
    ) -> Optional[Dict]:
        """
        투자설명서 검색 (상장일 기준 년도로 검색)
        
        Args:
            company_name: 회사명
            listing_date: 상장일 (YYYY-MM-DD 또는 MM/DD/YYYY 형식)
            
        Returns:
            검색 결과 딕셔너리 {'rcept_no': ..., 'report_nm': ..., 'date': ...}
        """
        # 상장일에서 년도 추출 (다양한 형식 지원)
        year = None
        try:
            if isinstance(listing_date, str):
                # YYYY-MM-DD 형식
                if '-' in listing_date and len(listing_date) >= 10:
                    year = listing_date[:4]
                # MM/DD/YYYY 형식
                elif '/' in listing_date:
                    parts = listing_date.split('/')
                    if len(parts) == 3:
                        year = parts[2][:4]  # YYYY 부분
                else:
                    year = listing_date[:4]
            else:
                year = str(listing_date.year)
        except:
            pass
        
        if not year or not year.isdigit() or len(year) != 4:
            year = str(datetime.now().year)
        
        # 종목명 정제 (구.XXX 등 제거)
        search_name = self._clean_company_name(company_name)
        
        # 검색 기간: 상장년도 전체
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        print(f"   [크롤링] {search_name} 검색 ({year}년)")
        
        url = "https://dart.fss.or.kr/dsab007/detailSearch.ax"
        
        # 검색 요청
        data = {
            'currentPage': '1',
            'maxResults': '100',
            'sort': 'date',
            'series': 'desc',
            'textCrpNm': search_name,
            'startDate': start_date,
            'endDate': end_date,
        }
        
        try:
            resp = self.session.post(url, data=data, timeout=self.timeout)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            rows = soup.select('table tbody tr')
            results = []
            
            for row in rows:
                cells = row.select('td')
                if len(cells) >= 4:
                    # 회사명: 코스닥/코스피 마크 제외하고 실제 회사명만 추출
                    corp_cell = cells[1]
                    # 마크 span 제거 후 a 태그 텍스트 또는 전체 텍스트
                    corp_link = corp_cell.select_one('a')
                    if corp_link:
                        corp = corp_link.get_text(strip=True)
                    else:
                        # span.tagCom_ 클래스 제거
                        for span in corp_cell.select('span[class^="tagCom"]'):
                            span.decompose()
                        corp = corp_cell.get_text(strip=True)
                    
                    report = cells[2].get_text(strip=True)
                    date = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                    
                    link = cells[2].select_one('a')
                    if link:
                        onclick = link.get('onclick', '') or ''
                        href = link.get('href', '') or ''
                        
                        # 접수번호 추출
                        match = re.search(r"'(\d{14})'", onclick)
                        if not match:
                            match = re.search(r'rcpNo=(\d+)', href)
                        
                        if match:
                            results.append({
                                'corp': corp,
                                'report_nm': report,
                                'date': date,
                                'rcept_no': match.group(1),
                            })
            
            # 회사명 필터링 (정확히 일치하거나 검색어를 포함하는 경우만)
            search_name_clean = search_name.replace(' ', '')
            filtered_results = [
                r for r in results 
                if search_name_clean in r['corp'].replace(' ', '') or r['corp'].replace(' ', '') in search_name_clean
            ]
            
            # 필터링 결과가 없으면 원본 사용 (부분 일치라도 허용)
            if not filtered_results:
                filtered_results = results
            
            # 1순위: 투자설명서
            prospectus = [r for r in filtered_results if '투자설명서' in r['report_nm']]
            if prospectus:
                print(f"   [크롤링] 투자설명서 발견: {prospectus[0]['rcept_no']}")
                return prospectus[0]
            
            # 2순위: 증권신고서(지분증권)
            securities = [r for r in filtered_results if '증권신고서' in r['report_nm'] and '지분증권' in r['report_nm']]
            if securities:
                print(f"   [크롤링] 증권신고서 발견: {securities[0]['rcept_no']}")
                return securities[0]
            
            print(f"   [크롤링] 검색 결과 없음")
            return None
            
        except Exception as e:
            print(f"   [크롤링] 검색 오류: {e}")
            return None
    
    def extract_dcm_no(self, rcept_no: str) -> Optional[str]:
        """
        DART 페이지에서 dcmNo 추출 (JavaScript 파싱)
        
        Args:
            rcept_no: 접수번호
            
        Returns:
            dcmNo 또는 None
        """
        url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
        
        try:
            resp = self.session.get(url, timeout=self.timeout)
            
            # JavaScript에서 dcmNo 추출
            # 패턴: node1['dcmNo'] = "6237363";
            pattern = r"node\d+\['dcmNo'\]\s*=\s*['\"](\d+)['\"]"
            matches = re.findall(pattern, resp.text)
            
            if matches:
                return matches[0]  # 첫 번째 dcmNo (본문)
            
            return None
            
        except Exception as e:
            print(f"   [크롤링] dcmNo 추출 오류: {e}")
            return None
    
    def download_pdf(
        self,
        rcept_no: str,
        dcm_no: str,
        save_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        PDF 직접 다운로드
        
        Args:
            rcept_no: 접수번호
            dcm_no: 문서번호
            save_dir: 저장 디렉토리 (기본: 캐시)
            
        Returns:
            다운로드된 PDF 경로 또는 None
        """
        if save_dir is None:
            save_dir = self.pdf_cache_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시 확인
        pdf_path = save_dir / f"{rcept_no}.pdf"
        if pdf_path.exists():
            print(f"   [캐시] PDF 캐시 사용: {pdf_path.name}")
            return pdf_path
        
        url = f"https://dart.fss.or.kr/pdf/download/pdf.do?rcp_no={rcept_no}&dcm_no={dcm_no}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': f'https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}',
        }
        
        try:
            resp = self.session.get(url, headers=headers, timeout=60)
            
            # PDF 확인
            if resp.content[:4] == b'%PDF':
                with open(pdf_path, 'wb') as f:
                    f.write(resp.content)
                print(f"   [다운로드] PDF 저장: {len(resp.content)/1024:.1f}KB")
                return pdf_path
            else:
                print(f"   [오류] PDF가 아닌 응답")
                return None
                
        except Exception as e:
            print(f"   [크롤링] PDF 다운로드 오류: {e}")
            return None
    
    def get_prospectus_pdf(
        self,
        company_name: str,
        listing_date: str,
    ) -> Optional[Path]:
        """
        투자설명서 PDF 가져오기 (검색 → dcmNo 추출 → 다운로드)
        
        Args:
            company_name: 회사명
            listing_date: 상장일 (YYYY-MM-DD)
            
        Returns:
            PDF 파일 경로 또는 None
        """
        # 1. 투자설명서 검색
        report = self.search_prospectus(company_name, listing_date)
        if not report:
            return None
        
        rcept_no = report['rcept_no']
        
        # 2. dcmNo 추출
        dcm_no = self.extract_dcm_no(rcept_no)
        if not dcm_no:
            print(f"   [오류] dcmNo 추출 실패")
            return None
        
        # 3. PDF 다운로드
        pdf_path = self.download_pdf(rcept_no, dcm_no)
        
        return pdf_path


def test_crawler():
    """크롤러 테스트"""
    crawler = DartCrawler()
    
    test_cases = [
        ("아이큐어", "2018-07-12"),
        ("파워넷", "2018-07-05"),
        ("모린스", "2009-09-25"),
        ("게임빌", "2009-07-30"),
    ]
    
    print("="*60)
    print("DART 크롤러 테스트")
    print("="*60)
    
    for name, date in test_cases:
        print(f"\n▶ {name} (상장일: {date})")
        pdf_path = crawler.get_prospectus_pdf(name, date)
        
        if pdf_path:
            print(f"   ✅ 성공: {pdf_path}")
        else:
            print(f"   ❌ 실패")


if __name__ == "__main__":
    test_crawler()
