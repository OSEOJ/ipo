"""
OpenDART API 연동 모듈
증권신고서 검색 및 문서 다운로드 기능 제공
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import io
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from ..config import DART_API_KEY, OPENDART_BASE_URL, CACHE_DIR


class DartAPI:
    """OpenDART API 클라이언트"""
    
    def __init__(self, api_key: str = DART_API_KEY):
        self.api_key = api_key
        self.base_url = OPENDART_BASE_URL
        
        # 세션 생성 - 프록시 우회, 타임아웃, 재시도 설정
        self.session = requests.Session()
        self.session.trust_env = False  # 프록시 비활성화
        self.timeout = 30  # 30초 타임아웃
        
        # 재시도 설정: 1회 재시도, 재시도 간 1초 대기
        retry_strategy = Retry(
            total=1,  # 최대 1회 재시도
            backoff_factor=1,  # 재시도 간 1초 대기
            status_forcelist=[500, 502, 503, 504],  # 서버 에러 시 재시도
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        
    def _request(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        """API 요청 공통 메서드"""
        params["crtfc_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response

    
    def search_reports(
        self,
        start_date: str,
        end_date: str,
        corp_code: Optional[str] = None,
        pblntf_ty: str = "",  # 빈 값: 전체, A: 정기공시, B: 주요사항, C: 발행공시
        page_no: int = 1,
        page_count: int = 100,
    ) -> Dict[str, Any]:
        """
        공시 목록 검색
        
        Args:
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            corp_code: 고유번호 (옵션)
            pblntf_ty: 공시유형 (A:정기공시)
            page_no: 페이지 번호
            page_count: 페이지당 개수
            
        Returns:
            검색 결과 딕셔너리
        """
        params = {
            "bgn_de": start_date.replace("-", ""),
            "end_de": end_date.replace("-", ""),
            "page_no": page_no,
            "page_count": page_count,
        }
        if pblntf_ty:
            params["pblntf_ty"] = pblntf_ty
        if corp_code:
            params["corp_code"] = corp_code
            
        response = self._request("list.json", params)
        return response.json()
    
    def search_securities_registration(
        self,
        start_date: str,
        end_date: str,
        corp_code: Optional[str] = None,
        page_no: int = 1,
        page_count: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        증권신고서(지분증권) 검색
        
        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            
        Returns:
            증권신고서 목록
        """
        # 공시유형 A (정기공시) + 보고서명 필터링
        result = self.search_reports(
            start_date=start_date,
            end_date=end_date,
            corp_code=corp_code,
            pblntf_ty="",
            page_no=page_no,
            page_count=page_count,
        )
        
        if result.get("status") != "000":
            print(f"API Error: {result.get('message')}")
            return []
        
        # 투자설명서 우선 필터링 (투자위험요소는 투자설명서에 있음)
        # 증권신고서에는 투자위험요소 섹션이 없음
        reports = result.get("list", [])
        
        # 1순위: 투자설명서
        prospectus = [
            r for r in reports 
            if "투자설명서" in r.get("report_nm", "")
        ]
        
        # 2순위: 증권신고서(지분증권) - fallback
        securities_reg = [
            r for r in reports 
            if ("증권신고서" in r.get("report_nm", "") and "지분증권" in r.get("report_nm", ""))
        ]
        
        # 투자설명서 우선, 없으면 증권신고서
        return prospectus if prospectus else securities_reg
    
    def download_document(self, rcept_no: str, save_dir: Optional[Path] = None) -> Optional[Path]:
        """
        공시서류 원본 다운로드 (ZIP) - 임시 폴더에 저장
        
        Args:
            rcept_no: 접수번호
            save_dir: 저장 디렉토리 (기본값: 임시 폴더)
            
        Returns:
            저장된 파일 경로
        """
        # 임시 폴더 사용 (캐시 시스템이 있으므로 영구 저장 불필요)
        if save_dir is None:
            save_dir = Path(tempfile.gettempdir()) / "dart_downloads"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        params = {"rcept_no": rcept_no}
        
        try:
            response = self._request("document.xml", params)
            
            # ZIP 파일 저장
            zip_path = save_dir / f"{rcept_no}.zip"
            with open(zip_path, "wb") as f:
                f.write(response.content)
            
            print(f"Downloaded: {zip_path}")
            return zip_path
            
        except Exception as e:
            print(f"Download failed for {rcept_no}: {e}")
            return None
    
    def extract_zip(self, zip_path: Path) -> List[Path]:
        """
        ZIP 파일 압축 해제
        
        Args:
            zip_path: ZIP 파일 경로
            
        Returns:
            추출된 파일 경로 목록
        """
        extract_dir = zip_path.parent / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                # 한글 파일명 처리
                try:
                    decoded_name = name.encode('cp437').decode('cp949')
                except (UnicodeDecodeError, UnicodeEncodeError):
                    decoded_name = name
                    
                extracted_path = extract_dir / decoded_name
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                
                with zf.open(name) as src, open(extracted_path, "wb") as dst:
                    dst.write(src.read())
                
                extracted_files.append(extracted_path)
        
        return extracted_files


    def get_company_info(self, corp_code: str) -> Optional[Dict[str, Any]]:
        """
        기업개황 조회
        API: https://opendart.fss.or.kr/api/company.json

        Args:
            corp_code: 고유번호 (8자리)

        Returns:
            기업 정보 딕셔너리:
            - corp_name: 정식 명칭
            - est_dt: 설립일 (YYYYMMDD)
            - induty_code: 업종코드
            - adres: 주소
            - 등
        """
        try:
            params = {"corp_code": corp_code}
            response = self._request("company.json", params)
            result = response.json()

            if result.get("status") == "000":
                return result
            else:
                print(f"[WARNING] 기업개황 조회 실패 ({corp_code}): {result.get('message')}")
                return None

        except Exception as e:
            print(f"[ERROR] 기업개황 조회 오류 ({corp_code}): {e}")
            return None

    def get_establishment_date(self, corp_code: str) -> Optional[str]:
        """
        설립일 조회

        Args:
            corp_code: 고유번호 (8자리)

        Returns:
            설립일 (YYYY-MM-DD 형식) 또는 None
        """
        info = self.get_company_info(corp_code)
        if info and info.get("est_dt"):
            est_dt = info["est_dt"]
            # YYYYMMDD → YYYY-MM-DD
            if len(est_dt) == 8:
                return f"{est_dt[:4]}-{est_dt[4:6]}-{est_dt[6:8]}"
        return None

    def get_corp_code_map(self) -> Dict[str, str]:
        """
        종목코드(6자리) -> 고유번호(8자리) 매핑 딕셔너리 반환
        API: https://opendart.fss.or.kr/api/corpCode.xml
        """
        try:
            import xml.etree.ElementTree as ET
            
            # 매핑 파일 다운로드 (캐싱)
            map_file = CACHE_DIR / "corpCode.zip"
            if not map_file.exists():
                params = {"crtfc_key": self.api_key}
                url = f"{self.base_url}/corpCode.xml"
                response = requests.get(url, params=params)
                
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(map_file, "wb") as f:
                    f.write(response.content)
            
            # 압축 해제 및 파싱
            mapping = {}
            with zipfile.ZipFile(map_file, "r") as zf:
                with zf.open("CORPCODE.xml") as xml_file:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    for child in root:
                        stock_code_elem = child.find("stock_code")
                        corp_code_elem = child.find("corp_code")

                        # 요소 존재 확인
                        if stock_code_elem is None or corp_code_elem is None:
                            continue

                        stock_code = stock_code_elem.text
                        corp_code = corp_code_elem.text

                        if stock_code and stock_code.strip():
                            mapping[stock_code.strip()] = corp_code.strip()
                            
            print(f"[INFO] Loaded corp_code map: {len(mapping)} companies")
            return mapping
            
        except Exception as e:
            print(f"[ERROR] Failed to load corp_code map: {e}")
            return {}


def test_api_connection():
    """API 연결 테스트"""
    api = DartAPI()
    
    # 매핑 테스트
    print("Loading corp_code map...")
    code_map = api.get_corp_code_map()
    print(f"Total mapped companies: {len(code_map)}")
    
    # 삼성전자 테스트
    if "005930" in code_map:
        print(f"Samsung Electronics corp_code: {code_map['005930']}")
    
    # 최근 1주일 증권신고서 검색
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"검색 기간: {start_date} ~ {end_date}")
    
    # 일반 공시 검색 테스트
    result = api.search_reports(start_date, end_date)
    print(f"API 상태: {result.get('status')}")
    print(f"메시지: {result.get('message')}")
    
    if result.get("status") == "000":
        print(f"총 {result.get('total_count', 0)}건 검색됨")
        
        # 증권신고서 필터링
        reports = result.get("list", [])
        sec_reports = [r for r in reports if "증권신고서" in r.get("report_nm", "")]
        print(f"증권신고서: {len(sec_reports)}건")
        
        for r in sec_reports[:5]:
            print(f"  - {r.get('corp_name')}: {r.get('report_nm')} ({r.get('rcept_no')})")
            
    return result


if __name__ == "__main__":
    test_api_connection()
