import numpy as np

# 파일 경로
file_path = 'sample_data/sample_true.npz'


# 파일 열기 및 데이터 확인
def inspect_npz_file(file_path):
    try:
        # npz 파일 로드
        data = np.load(file_path)

        # 키 및 데이터 정보 출력
        print(f"파일 경로: {file_path}")
        print(f"파일에 포함된 데이터 키: {list(data.keys())}")

        for key in data.keys():
            array = data[key]
            print(f"\n키: {key}")
            print(f" - 데이터 크기 (shape): {array.shape}")
            print(f" - 데이터 타입 (dtype): {array.dtype}")
            print(f" - 데이터 샘플 (일부 값): {array.ravel()[:5]}")  # 데이터 일부 출력

    except Exception as e:
        print(f"파일 확인 중 오류 발생: {e}")


# 함수 실행
inspect_npz_file(file_path)
