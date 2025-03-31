import skimage
import numpy as np
import cv2 as cv
import time
from skimage.segmentation import felzenszwalb

# 샘플 이미지(coffee) 불러오기
coffee = skimage.data.coffee()

# 처리 시간 측정을 위한 시작 시간 기록
start = time.time()

#########################################################
#skimage.future.graph를 더 이상 제공하지 않음
# RAG (Region Adjacency Graph) 생성
# 각 슈퍼픽셀의 평균 색상을 기반으로 유사도 그래프 생성
#g = skimage.future.graph.rag_mean_color(coffee, slic, mode='similarity')

# 정규화 그래프 컷(Normalized Cut)을 통해 영역 분할
# 슈퍼픽셀 간 유사도를 기반으로 더 큰 영역으로 병합
#ncut = skimage.future.graph.cut_normalized(slic, g)  # 정규화 절단
##########################################################

# 새로운 영상 분할 방식(felzenszwalb) 사용
# scale: 세그먼트 크기 설정, sigma: 블러 적용 강도
segments = felzenszwalb(coffee, scale=100, sigma=0.5, min_size=50)

# 처리 시간 출력
print(coffee.shape, ' Coffee 영상을 분할하는 데 ', time.time() - start, '초 소요')

# 분할된 결과에 경계선을 시각화
marking = skimage.segmentation.mark_boundaries(coffee, segments)

# float64(0~1) 범위의 이미지를 uint8(0~255)로 변환하여 OpenCV에서 표시 가능하게 함
segmented_coffee = np.uint8(marking * 255.0)

# 결과 출력 (RGB → BGR 변환 필요)
cv.imshow('Segmented Coffee', cv.cvtColor(segmented_coffee, cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()
