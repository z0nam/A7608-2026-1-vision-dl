# Context

## 작업 목적

- 과목: A7608 컴퓨터비전
- 범위: Chapter 3.2 비지도학습
- 날짜 기준 작업 세션: 2026-03-29
- 최종 산출물 목표: LaTeX beamer 발표 자료

## 기준 파일

- 세미나에서 함께 보는 기준 파일은 `chap03/slides/python_3장_seminar.ipynb`다.
- 발표용 노트북은 `3.2 비지도학습`부터 시작하도록 앞부분(`3.1`) 셀을 제거한 상태다.
- 원본 교재 실습 노트북 `chap03/python_3장.ipynb`, `chap03/colab_3장.ipynb`는 그대로 보관한다.
- `chap03/code/`는 노트북 대체물이 아니라 그림 재생성용 보조 스크립트다.

## 현재 상태

- `chap03/book/processed/`에는 전처리된 기준 문서 21장이 있다.
- 파일 순서 `00`부터 `20`까지가 실제 페이지 순서이며, 전부 3.2 관련 내용이다.
- 원본 스캔 JPG는 `chap03/book/raw/`에 있다.
- 저작권 이슈 때문에 `chap03/book/` 전체를 루트 `.gitignore`에 제외했다.
- OCR 결과는 `chap03/book/ocr/` 아래 페이지별 `.txt`로 생성되어 있다.
- OCR 인덱스는 `chap03/book/page_index.md`에 있다.
- 발표 초안은 `chap03/slides/main.tex`에 있다.

## book 전처리 메모

- `book` 관련 자료는 저장소 공개 기준 문서가 아니라 로컬 비공개 작업 자료로만 취급한다.
- 전처리본 21장은 발표 흐름을 잡는 데는 쓸 수 있지만, 상태가 아주 깔끔하다고 보기는 어렵다.
- 일부 페이지는 회전 보정, 여백, 대비, OCR 문장 단위가 아직 매끄럽지 않다.
- `ocr_doc_pages.py`에서 몇 장은 `PAGE_OVERRIDES`로 강제 회전과 `psm` 값을 따로 주고 있다.
- 즉, 현재 `book/processed`와 `book/ocr`는 “발표 정리용 참고본” 수준이고, 슬라이드 문장은 사람이 다시 다듬는 전제를 유지해야 한다.

## 구현된 작업

- OCR 스크립트
  - 파일: `chap03/tools/ocr_doc_pages.py`
  - 현재는 `chap03/book/processed/*.jpg`를 직접 OCR 대상으로 사용한다.
- 실습 재현 그림
  - `chap03/code/kmeans_elbow.py`
  - `chap03/code/pca_dbscan.py`
- 생성된 그림
  - `chap03/slides/figures/kmeans_elbow.png`
  - `chap03/slides/figures/pca_dbscan_min3.png`
  - `chap03/slides/figures/pca_dbscan_min50.png`
  - `chap03/slides/figures/pca_dbscan_min100.png`

## 발표 메모

1. 비지도학습 개요
2. K-means 개념과 엘보우 메서드
3. PCA로 차원 축소하는 이유
4. PCA 공간에서 DBSCAN 결과 비교
5. 실습 시연과 파라미터 해석

설명용으로 남길 교과서 도식 후보:
- KNN 반경/거리 개념도
- 결측치 보간 예시
- PCA 결과 표

실습으로 대체할 그림:
- `kmeans_elbow.png`: 군집 수 선택용 엘보우 메서드
- `pca_dbscan_min3.png`: PCA 2차원 투영 후 DBSCAN 기본 설정
- `pca_dbscan_min50.png`: `min_samples` 증가 시 군집 구조 변화
- `pca_dbscan_min100.png`: 밀도 조건 강화 시 노이즈 증가

발표 흐름 메모:
- K-means: 범주형 원-핫 인코딩, Min-Max 정규화, 엘보우 곡선
- PCA: 고차원 데이터를 2차원으로 줄여 구조를 읽기 쉽게 만듦
- PCA + DBSCAN: `min_samples` 변화에 따라 군집 수와 노이즈가 달라짐

## 다음 세션에서 바로 할 일

1. OCR 텍스트 21장을 읽되, 전처리/OCR 품질이 불안한 페이지는 원본 이미지와 함께 다시 확인한다.
2. `slides/main.tex`를 실제 발표 분량에 맞게 확장한다.
3. 교과서 도식 중 슬라이드에 남길 페이지와, 실습 결과로 대체할 그림을 최종 확정한다.
4. 필요하면 `code/` 아래 실습 스크립트를 더 잘게 쪼개 발표 파트별 파일로 분리한다.

## 실행 명령

- `python chap03/tools/ocr_doc_pages.py`
- `python chap03/code/kmeans_elbow.py`
- `python chap03/code/pca_dbscan.py`

## 주의 사항

- `chap03/book/processed/`를 기준 문서로 유지한다.
- 원본 스캔은 git에 올리지 않는다.
- OCR 품질은 페이지마다 편차가 있으므로 발표용 문장은 사람이 다시 다듬어야 한다.
- `README.md`에는 저작권 문제 때문에 `book` 관련 운영 메모를 쓰지 않고, 이런 내용은 `context.md`에만 남긴다.
