import cv2
import numpy as np

# 이미지 파일 경로
image_path = '/home/mila/j/jaewoo.lee/projects/text_prompt_sam/clipseg/waterbird_results/109.American_Redstart/American_Redstart_0020_104027.jpg'

# 이미지 읽기
image = cv2.imread(image_path)

# 각 채널의 최소값과 최대값 계산
min_val = np.min(image, axis=(0, 1))
max_val = np.max(image, axis=(0, 1))

# 결과 출력
print(f"Minimum values per channel: {min_val}")
print(f"Maximum values per channel: {max_val}")
