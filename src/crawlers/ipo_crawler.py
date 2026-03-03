"""
38.co.kr IPO 기본 정보 크롤러
종목명, 종목코드, 시장구분, 업종, 상장일만 추출

requests + BeautifulSoup 기반 (Selenium 불필요)
캐싱 지원: 종목별 JSON 파일로 저장하여 재실행 시 속도 향상
curl fallback: SSL 문제 시 curl subprocess 사용
"""

import os
import re
import json
import time
import requests
import urllib3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from tqdm import tqdm

from ..config import IPO_CACHE_DIR
from ..market_data import MarketDataFetcher
# Future structure imports
from ..dart.crawler import DartCrawler
from ..extractors.pdf import PDFParser

from .base import fetch_with_curl
from .parsers import (
    parse_shares_from_text,
    parse_rate_from_text,
    parse_percentage_from_text,
    parse_date,
    parse_listing_date
)

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class IPOCrawler:
    """38.co.kr IPO 기본 정보 크롤러"""

    BASE_URL = "https://www.38.co.kr"
    LIST_URL = f"{BASE_URL}/html/fund/index.htm"

    def __init__(self):
        """IPO 크롤러 초기화"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        })

        # 환경변수에서 프록시 설정 (있으면 사용)
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        if http_proxy or https_proxy:
            self.session.proxies = {
                'http': http_proxy,
                'https': https_proxy,
            }
            print(f"[INFO] 프록시 사용: {https_proxy or http_proxy}")
        
        # 시장 데이터 fetcher (코스닥 수익률용)
        self.market_fetcher = MarketDataFetcher()
        
        # DART API (업력 계산용)
        self.dart_api = None
        self.corp_code_map = None
        
        # for PDF Extraction (Fallback)
        self.dart_crawler = None
        self.pdf_parser = None

    def _get_cache_path(self, detail_no: str) -> Path:
        """캐시 파일 경로 반환 (detail_no 기준)"""
        return IPO_CACHE_DIR / f"{detail_no}.json"

    def _load_from_cache(self, detail_no: str) -> Optional[Dict]:
        """캐시에서 IPO 데이터 로드"""
        cache_path = self._get_cache_path(detail_no)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _save_to_cache(self, detail_no: str, data: Dict):
        """IPO 데이터를 캐시에 저장"""
        cache_path = self._get_cache_path(detail_no)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[WARNING] 캐시 저장 실패 ({detail_no}): {e}")

    def _is_spac(self, stock_name: str) -> bool:
        """스팩(SPAC) 종목 여부 확인"""
        spac_keywords = [
            '스팩', 'SPAC', 'Spac',
            '제1호', '제2호', '제3호', '제4호', '제5호',
            '제6호', '제7호', '제8호', '제9호',
            '기업인수목적'
        ]
        return any(keyword in stock_name for keyword in spac_keywords)

    def start(self):
        """세션 시작 (호환성 유지)"""
        pass

    def stop(self):
        """세션 종료 (호환성 유지)"""
        self.session.close()
    
    def _init_dart_api(self):
        """DART API 초기화 (지연 로딩)"""
        if self.dart_api is None:
            from ..dart.api import DartAPI
            self.dart_api = DartAPI()
            print("[INFO] DART API 초기화 중...")
            self.corp_code_map = self.dart_api.get_corp_code_map()
    
    def _get_company_age(self, stock_code: str, listing_date: str) -> Optional[float]:
        """업력 계산 (설립일 ~ 상장일)"""
        try:
            self._init_dart_api()
            
            corp_code = self.corp_code_map.get(stock_code)
            if not corp_code:
                return None
            
            est_date = self.dart_api.get_establishment_date(corp_code)
            if not est_date:
                return None
            
            # 날짜 파싱 (다양한 형식 지원)
            est_dt = parse_listing_date(est_date)
            list_dt = parse_listing_date(listing_date)
            
            if not est_dt or not list_dt:
                return None
            
            # 업력 계산 (년 단위)
            age = round((list_dt - est_dt).days / 365, 1)
            return age if age > 0 else None
            
        except Exception as e:
            return None
    
    def _get_page(self, page_num: int = 1) -> Optional[BeautifulSoup]:
        """특정 페이지 HTML 가져오기 (requests 실패 시 curl fallback)"""
        url = f"{self.LIST_URL}?o=nw&page={page_num}"

        # 1. requests 시도
        try:
            response = self.session.get(url, timeout=30, verify=False)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            return BeautifulSoup(response.text, 'lxml')
        except requests.exceptions.RequestException as e:
            # 2. curl fallback
            html = fetch_with_curl(url)
            if html:
                return BeautifulSoup(html, 'lxml')
            print(f"[ERROR] 페이지 {page_num} 요청 실패: {e}")
            return None

    def _parse_list_page(self, soup: BeautifulSoup) -> List[Dict]:
        """목록 페이지에서 IPO 기본 정보 추출"""
        items = []
        seen_detail_nos = set()

        for table in soup.find_all('table'):
            for row in table.find_all('tr'):
                cols = row.find_all('td')

                if len(cols) < 7:
                    continue

                # 종목명 및 detail_no 추출
                link_tag = cols[0].find('a')
                if not link_tag or 'o=v' not in link_tag.get('href', ''):
                    continue

                href = link_tag.get('href', '')
                no_match = re.search(r'no=(\d+)', href)
                if not no_match:
                    continue

                detail_no = no_match.group(1)

                # 중복 체크
                if detail_no in seen_detail_nos:
                    continue
                seen_detail_nos.add(detail_no)

                # 종목명
                stock_name = link_tag.get_text(strip=True)

                # 상장일
                listing_date_str = cols[1].get_text(strip=True)
                listing_date = parse_date(listing_date_str)

                if not listing_date:
                    continue

                # 시장구분 (종목명에서 1차 추출)
                market_type = None
                if '(유가)' in stock_name or '(코스피)' in stock_name:
                    market_type = 'KOSPI'
                elif '(코넥스)' in stock_name:
                    market_type = 'KONEX'
                elif '(코스닥)' in stock_name:
                    market_type = 'KOSDAQ'

                item = {
                    '종목명': stock_name,
                    '상장일': listing_date,
                    'detail_no': detail_no,
                    'detail_url': urljoin(self.BASE_URL, href),
                    '시장구분': market_type # 1차 추출 결과 저장
                }

                items.append(item)

        return items

    def _get_detail_info(self, detail_no: str, initial_market_type: str = None) -> Dict:
        """상세 페이지에서 모든 IPO 정보 추출"""
        url = f"{self.BASE_URL}/html/fund/?o=v&no={detail_no}&l="

        result = {
            '종목코드': None,
            '시장구분': initial_market_type, # 초기값 설정
            '업종': None,
            '기관경쟁률': None,
            '의무보유확약': None,
            '기관배정': None,
            '개인경쟁률': None,
        }

        # 총 공모주 수 (기관배정 비율 계산용)
        total_shares = None

        try:
            # requests 시도
            try:
                response = self.session.get(url, timeout=30, verify=False)
                response.raise_for_status()
                response.encoding = response.apparent_encoding or 'utf-8'
                html = response.text
            except requests.exceptions.RequestException:
                # curl fallback
                html = fetch_with_curl(url)
                if not html:
                    return result

            soup = BeautifulSoup(html, 'lxml')

            # 테이블에서 직접 추출 (td 텍스트가 정확히 일치하는 경우)
            for td in soup.find_all('td'):
                td_text = td.get_text(strip=True)

                # 종목코드 추출
                if td_text == '종목코드':
                    next_td = td.find_next_sibling('td')
                    if next_td:
                        code_value = next_td.get_text(strip=True)
                        if re.match(r'^[A-Z0-9]{6}$', code_value):
                            result['종목코드'] = code_value

                # 시장구분 추출
                elif td_text == '시장구분':
                    next_td = td.find_next_sibling('td')
                    if next_td:
                        market_value = next_td.get_text(strip=True)
                        if '코스피' in market_value or 'KOSPI' in market_value.upper() or '유가증권' in market_value:
                            result['시장구분'] = 'KOSPI'
                        elif '코스닥' in market_value or 'KOSDAQ' in market_value.upper():
                            result['시장구분'] = 'KOSDAQ'
                        elif 'KRX' in market_value.upper() or '코넥스' in market_value:
                            result['시장구분'] = 'KONEX'

                # 업종 추출
                elif td_text == '업종':
                    next_td = td.find_next_sibling('td')
                    if next_td:
                        industry_value = next_td.get_text(strip=True)
                        industry_value = re.sub(r'\s+', ' ', industry_value).strip()
                        if len(industry_value) >= 2 and industry_value not in ['해당없음', '-', '']:
                            result['업종'] = industry_value

                # 기관경쟁률 추출
                elif td_text == '기관경쟁률':
                    next_td = td.find_next_sibling('td')
                    if next_td:
                        value = next_td.get_text(strip=True)
                        rate = parse_rate_from_text(value)
                        if rate is not None:
                            result['기관경쟁률'] = rate

                # 청약경쟁률 → 개인경쟁률로 사용
                elif td_text == '청약경쟁률':
                    next_td = td.find_next_sibling('td')
                    if next_td:
                        value = next_td.get_text(strip=True)
                        rate = parse_rate_from_text(value)
                        if rate is not None:
                            result['개인경쟁률'] = rate

                # 의무보유확약 추출 (여러 레이블명 지원)
                elif td_text in ['의무보유확약', '확약비율', '보호예수']:
                    next_td = td.find_next_sibling('td')
                    if next_td:
                        value = next_td.get_text(strip=True)
                        rate = parse_percentage_from_text(value)
                        if rate is not None:
                            result['의무보유확약'] = rate

                # 총 공모주식수 추출
                elif td_text in ['공모주식수', '총공모주식수', '공모주수', '공모수량']:
                    next_td = td.find_next_sibling('td')
                    if next_td:
                        value = next_td.get_text(strip=True)
                        shares = parse_shares_from_text(value)
                        if shares is not None:
                            total_shares = shares

            # 그룹별배정에서 기관투자자등 비율 추출
            inst_shares = None  # 기관배정 주 수

            for element in soup.find_all(string=re.compile(r'기관투자자')):
                # 해당 요소의 부모 행(tr) 찾기
                parent_row = element.find_parent('tr')
                if parent_row:
                    row_text = parent_row.get_text(strip=True)

                    # 1. 먼저 괄호 안의 퍼센트 찾기: "(75%)" 또는 "(65~75%)"
                    if '(' in row_text and '%' in row_text and ')' in row_text:
                        for cell in parent_row.find_all(['td', 'th']):
                            cell_text = cell.get_text(strip=True)
                            if '(' in cell_text and '%' in cell_text and ')' in cell_text:
                                start = cell_text.find('(')
                                end = cell_text.find(')')
                                if start < end:
                                    pct_str = cell_text[start+1:end].replace('%', '').strip()
                                    try:
                                        # 범위 형식 처리: "65~75" or "65-75"
                                        if '~' in pct_str or '-' in pct_str:
                                            separator = '~' if '~' in pct_str else '-'
                                            parts = pct_str.split(separator)
                                            if len(parts) == 2:
                                                low = float(parts[0].strip())
                                                high = float(parts[1].strip())
                                                rate = (low + high) / 2
                                            else:
                                                continue
                                        else:
                                            rate = float(pct_str)

                                        # 100이 아닌 값만 (100%는 청약증거금율일 가능성)
                                        if rate != 100:
                                            result['기관배정'] = rate
                                            break
                                    except ValueError:
                                        pass

                    # 2. 퍼센트 없으면 주 수 추출: "1,000,000 주" 또는 "800,000~1,000,000 주"
                    if result['기관배정'] is None:
                        for cell in parent_row.find_all(['td', 'th']):
                            cell_text = cell.get_text(strip=True)
                            if '주' in cell_text:
                                shares = parse_shares_from_text(cell_text)
                                if shares is not None and shares > 0:
                                    inst_shares = shares
                                    break

                    if result['기관배정'] is not None:
                        break

            # 3. 퍼센트가 없고 주 수만 있으면, 총 공모주 수로 비율 계산
            if result['기관배정'] is None and inst_shares is not None and total_shares is not None:
                if total_shares > 0:
                    result['기관배정'] = round((inst_shares / total_shares) * 100, 2)

            return result

        except requests.exceptions.RequestException as e:
            print(f"[WARNING] 상세 정보 추출 실패 (detail_no={detail_no}): {e}")
            return result

    def _get_stock_code_and_market(self, detail_no: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """상세 페이지에서 종목코드, 시장구분, 업종 추출"""
        info = self._get_detail_info(detail_no)
        return info['종목코드'], info['시장구분'], info['업종']

    def _get_total_pages(self, soup: BeautifulSoup) -> int:
        """총 페이지 수 확인 (페이지네이션 끝까지 탐색)"""
        # 1. 기존 방식으로 먼저 시도
        page_links = soup.find_all('a', href=re.compile(r'page=\d+'))

        max_page = 1
        for link in page_links:
            match = re.search(r'page=(\d+)', link.get('href', ''))
            if match:
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)

        # 2. "맨끝"/"마지막" 링크가 있는지 확인
        last_links = soup.find_all('a', string=re.compile(r'맨끝|마지막|끝|last', re.I))
        for link in last_links:
            href = link.get('href', '')
            match = re.search(r'page=(\d+)', href)
            if match:
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)

        # 3. 현재 페이지에서 보이는 가장 큰 번호 + 추정
        if max_page < 20: 
            print(f"   [경고] 감지된 페이지 수가 적음 ({max_page}페이지). 실제로는 더 많을 수 있습니다.")
            max_page = 100

        return max_page

    def crawl(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """IPO 기본 정보 크롤링"""
        print("\n" + "=" * 60)
        print("IPO 기본 정보 크롤링 시작")
        print("=" * 60)

        try:
            # 1단계: 목록 페이지에서 종목명, 상장일 수집
            print("\n[1단계] 목록 페이지 크롤링...")

            first_page_soup = self._get_page(1)
            if not first_page_soup:
                print("[ERROR] 첫 페이지 로드 실패")
                return []

            total_pages = self._get_total_pages(first_page_soup)
            print(f"총 {total_pages} 페이지 발견")

            all_items = []
            stop_crawling = False

            for page_num in range(1, total_pages + 1):
                if stop_crawling:
                    break

                print(f"페이지 {page_num}/{total_pages} 크롤링 중...")

                if page_num == 1:
                    soup = first_page_soup
                else:
                    time.sleep(1.0)
                    soup = self._get_page(page_num)

                if not soup:
                    continue

                items = self._parse_list_page(soup)

                if not items:
                    print(f"   페이지 {page_num}에 데이터 없음. 크롤링 종료.")
                    break

                for item in items:
                    listing_date = item.get('상장일')

                    # 날짜 필터링
                    if listing_date:
                        if start_date and listing_date < start_date:
                            stop_crawling = True
                            break
                        if end_date and listing_date > end_date:
                            continue

                    all_items.append(item)

                    # 개수 제한
                    if limit and len(all_items) >= limit:
                        stop_crawling = True
                        break

            print(f"\n목록 수집 완료: {len(all_items)}개")

            # 2단계: 상세 페이지에서 종목코드, 시장구분, 업종 추출 (캐시 활용)
            print("\n[2단계] 상세 정보 수집 (캐시 활용)...")

            results = []
            cached_count = 0
            crawled_count = 0
            spac_count = 0
            kospi_count = 0

            for item in tqdm(all_items, desc="상세 정보"):
                detail_no = item.get('detail_no')
                listing_date_str = item['상장일'].strftime('%Y-%m-%d')
                stock_name = item['종목명']

                if not detail_no:
                    continue
                
                # 스팩 종목 제외
                if self._is_spac(stock_name):
                    spac_count += 1
                    continue

                # 캐시 먼저 확인
                cached_data = self._load_from_cache(detail_no)
                if cached_data:
                    if cached_data.get('종목코드'):
                        if self._is_spac(cached_data.get('종목명', '')):
                            spac_count += 1
                            continue
                        results.append(cached_data)
                        cached_count += 1
                        continue

                # 캐시 없으면 상세 페이지 크롤링
                detail_info = self._get_detail_info(detail_no, item.get('시장구분'))

                if not detail_info['종목코드']:
                    continue

                result = {
                    '종목명': stock_name,
                    '종목코드': detail_info['종목코드'],
                    '시장구분': detail_info['시장구분'] or '',
                    '업종': detail_info['업종'] or '',
                    '상장일': listing_date_str,
                    '기관경쟁률': detail_info['기관경쟁률'],
                    '의무보유확약': detail_info['의무보유확약'],
                    '기관배정': detail_info['기관배정'],
                    '개인경쟁률': detail_info['개인경쟁률'],
                }

                # 캐시에 저장
                self._save_to_cache(detail_no, result)
                results.append(result)
                crawled_count += 1

                time.sleep(0.5)

            print(f"\n크롤링 완료: {len(results)}개 종목 (캐시: {cached_count}, 신규: {crawled_count})")
            print(f"   제외: 스팩 {spac_count}개, 유가증권 {kospi_count}개")
            
            # [추가] PDF Fallback: 기관경쟁률 결측치 보완
            print("\n[2.5단계] 기관경쟁률 결측치 보완 (DART PDF 활용)...")
            fallback_count = 0
            
            for item in tqdm(results, desc="경쟁률 보완"):
                if not item.get('기관경쟁률') or item['기관경쟁률'] == 0:
                    stock_name = item['종목명']
                    listing_date = item['상장일']
                    
                    try:
                        if self.dart_crawler is None:
                            self.dart_crawler = DartCrawler()
                            self.pdf_parser = PDFParser()
                            
                        pdf_path = self.dart_crawler.get_prospectus_pdf(stock_name, listing_date)
                        
                        if pdf_path:
                            rate = self.pdf_parser.extract_competition_rate(pdf_path)
                            if rate:
                                item['기관경쟁률'] = rate
                                detail_no = next((x['detail_no'] for x in all_items if x['종목명'] == stock_name), None)
                                if detail_no:
                                    self._save_to_cache(detail_no, item)
                                fallback_count += 1
                    except Exception as e:
                        print(f"   [에러] {stock_name} 보완 중 오류: {e}")
                        
            print(f"   보완 완료: {fallback_count}개 종목")

            # 3단계: 업력 및 시장 지수 15일 수익률 계산
            print("\n[3단계] 업력 및 시장 지수 15일 수익률 계산...")
            
            market_return_cache = {}  # {(date, market_type): return}
            
            for item in tqdm(results, desc="업력/수익률"):
                listing_date = item['상장일']
                stock_code = item['종목코드']
                market_type = item.get('시장구분', 'KOSDAQ')
                if not market_type: market_type = 'KOSDAQ'  # 기본값
                
                if '업력' not in item or item.get('업력') is None:
                    item['업력'] = self._get_company_age(stock_code, listing_date)
                
                # 시장지수 15일 수익률 (기존 '코스닥_15일_수익률' 대체)
                if '시장지수_15일_수익률' not in item:
                    cache_key = (listing_date, market_type)
                    if cache_key not in market_return_cache:
                        market_return_cache[cache_key] = self.market_fetcher.get_market_return(listing_date, market_type)
                    item['시장지수_15일_수익률'] = market_return_cache[cache_key]
            
            age_count = sum(1 for r in results if r.get('업력') is not None)
            market_return_count = sum(1 for r in results if r.get('시장지수_15일_수익률') is not None)
            print(f"   업력 계산 성공: {age_count}/{len(results)}")
            print(f"   시장 수익률 계산 성공: {market_return_count}/{len(results)}")
            
            return results

        finally:
            self.stop()
