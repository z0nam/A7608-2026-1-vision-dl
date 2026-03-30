## <딥러닝 파이토치 교과서>(길벗, 2022) 도서의 예제 소스 파일입니다.

∙Jupyter Notebook과 Colab용 파일을 제공합니다. </br>

∙실습에 필요한 데이터셋 대부분은 챕터별 data 폴더에 들어 있습니다.</br>

### =========================================

### ∙아래 설명을 꼭 읽어주세요!!!
∙대용량 데이터셋은 깃허브에 올라가지 않습니다. 따라서 아래의 안내에 따라 데이터를 내려받은 후 각 폴더에 넣어 사용합니다.

∙1) 10장 wiki.ko.vec만 별도로 내려받습니다.(약 2GB))</br>
https://fasttext.cc/docs/en/pretrained-vectors.html 에서 Korea의 txt 파일 </br>

∙내려받은 wiki.ko.vec 파일은 10장의 data 폴더 > wiki.co 파일에 넣어서 사용합니다.

∙ 2) 아래 링크에서 glove.6b.100d.txt를 내려받습니다. 10장의 data 폴더에 넣어 실습하세요. </br>

∙ 10장 [glove.6B.100d.txt_내려받기](https://github.com/gilbutITbook/080263/releases/download/0.2/glove.6B.100d.txt) </br>
   사이트 URL: https://nlp.stanford.edu/projects/glove/ </br>

∙데이터셋은 python과 colab 모두 동일하게 사용하며, 일부 압축 파일은 colab 실습용입니다.</br>

∙8장부터는 colab에서 '런타임 유형'을 'GPU'로 설정하고 실습하는 것이 좋습니다. (실행 시간 단축)</br>

### =========================================
∙colab에 필요한 데이터는 파일을 업로드해서 사용해야 합니다. 두 가지 방식이 있는데, 책에서는 대부분 PC에서 파일을 업로드하는 방식으로 되어 있습니다. </br>
구글 드라이브를 사용 중이라면 드라이브에 마운트해 사용하는 것이 좀 더 편리합니다. (부록을 참고하세요)</br>

#PC에서 파일 업로드 </br>
from google.colab import files </br>
file_uploaded=files.upload() </br>

#구글 드라이브 마운트 </br>
from google.colab import drive </br>
drive.mount('/content/drive/') </br>

### =========================================
### 발표 작업 워크플로

이 저장소는 교과서 실습 코드뿐 아니라, 장별 발표 준비 작업도 함께 진행한다.

- 발표 작업은 해당 장 폴더 바로 아래에 필요한 폴더만 납작하게 만든다.
- 교과서 스캔본이 있으면 원본은 별도 보관 폴더로 옮기고, 전처리한 페이지를 기준 문서로 사용한다.
- 저작권 이슈가 있는 원본 스캔은 반드시 `.gitignore`에 추가한다.
- 발표에 쓰는 그림은 가능하면 교과서 캡처보다 실습 코드 출력으로 다시 만든다.
- 다른 세션에서 이어서 작업할 수 있도록 장별 `context.md`를 유지한다.

권장 구조 예시는 다음과 같다.

- `chapXX/book/`: 원본 스캔, 전처리본, OCR 텍스트, 페이지 인덱스
- `chapXX/code/`: 발표용으로 다듬은 실습 코드
- `chapXX/tools/`: OCR 등 자료 정리용 처리 스크립트
- `chapXX/slides/`: 최종 LaTeX beamer 발표 자료
- `chapXX/slides/figures/`: 슬라이드에서 쓰는 그림
- `chapXX/context.md`: 다음 세션 인수인계 문서
