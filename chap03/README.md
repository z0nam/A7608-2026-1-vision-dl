# CH3 발표 작업 공간

이 폴더는 Chapter 3 발표 준비를 위한 작업 공간이다. 우선순위는 다음과 같다.

1. `book/processed/`의 전처리본 21장을 OCR로 텍스트화한다.
2. 3.2 비지도학습 설명에 필요한 핵심 그림을 정리한다.
3. 교과서 그림과 겹치는 항목은 가능한 한 실습 코드로 재현한다.
4. 최종 산출물은 `slides/` 아래 LaTeX beamer 발표 자료로 정리한다.

## 작업 원칙

- 세미나 참여자와 함께 보는 기준 파일은 기존 노트북(`python_3장.ipynb`, `colab_3장.ipynb`)이다.
- 노트북의 셀 순서와 섹션 구조는 가능하면 유지한다.
- `code/` 아래 파일은 발표 그림 재생성이나 빠른 실행을 위한 보조 스크립트로만 사용한다.
- 노트북 내용을 대체하는 별도 강의용 코드 구조로 키우지 않는다.

## 폴더 구조

- `book/`: 교과서 원본, 전처리본, OCR 텍스트, 페이지 인덱스
- `book/raw/`: 원본 스캔 21장
- `book/processed/`: 전처리 완료본 21장. 발표 작업의 기준 문서
- `book/ocr/`: 페이지별 OCR 텍스트
- `slides/figures/`: 슬라이드에서 직접 사용하는 그림
- `slides/`: 발표 슬라이드 초안
- `code/`: 발표 중 보여줄 실습 재현 코드
- `tools/`: OCR 같은 자료 정리용 처리 스크립트
- `context.md`: 다른 세션으로 넘길 때 필요한 작업 맥락

## 현재 정리 기준

- `book/processed/`의 21장은 모두 Chapter 3.2 관련 내용이며, 파일 번호 `00`부터 `20`까지가 페이지 순서다.
- 3.2 실습 연결 가능 항목은 `K-means`, `PCA + DBSCAN`이다.
- 교과서 페이지 인덱스는 `book/page_index.md`에 둔다.
- 실습에서 재현 가능한 그림은 `slides/figures/` 아래에 저장한다.

## 다음 작업

- `python tools/ocr_doc_pages.py`
- `python code/kmeans_elbow.py`
- `python code/pca_dbscan.py`
- OCR 결과를 바탕으로 `slides/main.tex`를 채운다.
- 세션을 넘길 때는 `context.md`를 먼저 갱신한다.
