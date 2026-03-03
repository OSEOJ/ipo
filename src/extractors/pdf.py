"""
PDF 파싱 모듈
DART 투자설명서 PDF에서 기관경쟁률 추출

PyMuPDF의 find_tables() 기능을 사용하여 표 구조를 정확히 파싱합니다.
"""
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional, List, Tuple

class PDFParser:
    """PDF 파싱 및 데이터 추출 클래스"""
    
    def __init__(self):
        # 경쟁률 관련 컬럼 이름들
        self.rate_column_names = ["단순경쟁률", "기관경쟁률", "경쟁률"]
        # 타겟 섹션 키워드
        self.section_keywords = ["공모가격", "수요예측", "모집 또는 매출에 관한 일반사항"]
        
    def extract_competition_rate(self, pdf_path: Path) -> Optional[float]:
        """
        PDF에서 기관경쟁률(또는 단순경쟁률) 추출
        
        전략:
        1. TOC에서 "공모가격 결정방법" 섹션 위치 파악
        2. 해당 페이지들에서 테이블 추출 (find_tables)
        3. 테이블에서 "경쟁률" 컬럼 찾아 값 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            경쟁률 (float) 또는 None
        """
        if not pdf_path or not pdf_path.exists():
            return None
            
        try:
            doc = fitz.open(pdf_path)
            
            # 1. TOC 기반 타겟 페이지 범위 결정
            target_pages = self._find_target_pages_from_toc(doc.get_toc(), doc.page_count)
            
            # 타겟 페이지가 없으면 전체 문서 검색
            if not target_pages:
                target_pages = list(range(doc.page_count))
            
            # 2. 테이블 기반 추출 시도
            rate = self._extract_from_tables(doc, target_pages)
            
            # 3. 테이블 추출 실패 시 텍스트 기반 폴백
            if rate is None:
                rate = self._extract_from_text(doc, target_pages)
            
            doc.close()
            return rate
            
        except Exception as e:
            print(f"   [PDF] 파싱 오류: {e}")
            return None

    def _find_target_pages_from_toc(self, toc: List, total_pages: int) -> List[int]:
        """TOC에서 '공모가격' 관련 섹션의 페이지 범위 추출"""
        if not toc:
            return []
            
        target_start = None
        target_end = None
        
        for i, entry in enumerate(toc):
            level, title, page_num = entry
            
            # "공모가격" 키워드가 있는 섹션 찾기
            if "공모가격" in title:
                target_start = page_num - 1  # 0-indexed
                
                # 다음 동일/상위 레벨 섹션까지 범위 설정
                for j in range(i + 1, len(toc)):
                    next_level, _, next_page = toc[j]
                    if next_level <= level:
                        target_end = next_page - 1
                        break
                break
        
        if target_start is not None:
            if target_end is None:
                target_end = min(target_start + 20, total_pages)  # 최대 20페이지
            return list(range(target_start, target_end))
            
        return []

    def _extract_from_tables(self, doc, pages: List[int]) -> Optional[float]:
        """
        PyMuPDF의 find_tables()를 사용하여 테이블에서 경쟁률 추출
        
        테이블 구조 예시:
        | 참여건수 | 신청수량 | 단순경쟁률 |
        | 506건 | 182,560,000주 | 478.41:1 |
        """
        for page_num in pages:
            if page_num >= doc.page_count:
                continue
                
            page = doc[page_num]
            
            # 페이지에 경쟁률 키워드가 없으면 스킵
            text = page.get_text()
            if not any(k in text for k in self.rate_column_names):
                continue
            
            # 테이블 추출
            try:
                tables = page.find_tables()
            except Exception:
                continue
                
            for table in tables.tables:
                rate = self._parse_rate_from_table(table)
                if rate is not None:
                    return rate
                    
        return None

    def _parse_rate_from_table(self, table) -> Optional[float]:
        """
        테이블 객체에서 경쟁률 값 추출
        
        Args:
            table: PyMuPDF Table 객체
            
        Returns:
            경쟁률 (float) 또는 None
        """
        try:
            # pandas DataFrame으로 변환
            df = table.to_pandas()
            
            if df.empty:
                return None
            
            # 컬럼명에서 경쟁률 컬럼 찾기
            rate_col_idx = None
            for col_name in df.columns:
                if any(k in str(col_name) for k in self.rate_column_names):
                    rate_col_idx = df.columns.get_loc(col_name)
                    break
            
            if rate_col_idx is None:
                # 첫 번째 행을 헤더로 확인
                first_row = df.iloc[0].tolist()
                for idx, cell in enumerate(first_row):
                    if any(k in str(cell) for k in self.rate_column_names):
                        rate_col_idx = idx
                        break
            
            if rate_col_idx is None:
                return None
            
            # 해당 컬럼의 값들에서 경쟁률 추출
            for idx, row in df.iterrows():
                cell_value = str(row.iloc[rate_col_idx]) if rate_col_idx < len(row) else ""
                
                rate = self._parse_rate_value(cell_value)
                if rate is not None:
                    return rate
                    
        except Exception:
            pass
            
        return None

    def _parse_rate_value(self, value: str) -> Optional[float]:
        """
        셀 값에서 경쟁률 숫자 추출
        
        지원 형식:
        - "478.41:1"
        - "478.41 : 1"
        - "478.41"
        """
        if not value or value.lower() in ['none', 'nan', '']:
            return None
            
        # 패턴 1: "X:1" 형식
        match = re.search(r'([\d,]+(?:\.\d+)?)\s*:\s*1', value)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        # 패턴 2: 소수점이 있는 숫자 (컬럼명으로 이미 경쟁률임을 확인함)
        match = re.search(r'([\d,]+\.\d+)', value)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass
                
        return None

    def _extract_from_text(self, doc, pages: List[int]) -> Optional[float]:
        """
        테이블 추출 실패 시 텍스트 기반 폴백
        
        "단순경쟁률" 또는 "기관경쟁률" 키워드 근처에서 "X:1" 패턴 검색
        """
        for page_num in pages:
            if page_num >= doc.page_count:
                continue
                
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                # 경쟁률 키워드가 있는 라인
                if any(k in line for k in self.rate_column_names):
                    # 현재 라인과 다음 몇 라인에서 X:1 패턴 검색
                    context = '\n'.join(lines[i:min(i+5, len(lines))])
                    
                    # "X:1" 패턴 검색
                    match = re.search(r'([\d,]+(?:\.\d+)?)\s*:\s*1', context)
                    if match:
                        try:
                            return float(match.group(1).replace(',', ''))
                        except ValueError:
                            pass
                            
        return None
