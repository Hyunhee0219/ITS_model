from data import *
from controller import *
import time
from tqdm import tqdm
import logging

def ranked_precision(recommended_list, relevant_items):
    """
    Ranked Precision 계산 함수.
    
    매개변수:
    recommended_list: 추천된 항목 리스트 (마할라노비스 거리를 기반으로 랭크됨)
    relevant_items: 실제 관련된 항목 (올바른 선수 학습 개념)
    
    반환값:
    ranked_precision_value: Ranked Precision 점수
    """
    relevant_count = 0  # 관련 항목 카운트
    precision_sum = 0  # Precision 합산
    
    # 추천 리스트를 순회하며 Precision 계산
    for k, item in enumerate(recommended_list, 1):  # k는 1부터 시작
        if item in relevant_items:  # 만약 추천된 항목이 관련 항목에 있다면
            relevant_count += 1
            precision_sum += relevant_count / k  # Precision 값 누적
    
    # 추천된 리스트의 길이로 정규화하여 Ranked Precision 계산
    ranked_precision_value = precision_sum / len(recommended_list) if recommended_list else 0
    
    # 결과를 정수로 변환
    return int(round(ranked_precision_value))

def rank_by_area(top_successor_recommendations_df, relevant_items):
    """
    영역별로 그룹화하고, 각 그룹 내에서 랭크를 매기고, 해당 영역에 대한 Ranked Precision을 표기하는 함수.
    """
    grouped_by_area = top_successor_recommendations_df.groupby('영역')

    for area, group in grouped_by_area:
        recommended_list = group['knowledgeTag'].tolist()
        ranked_precision_value = ranked_precision(recommended_list, relevant_items)
        
        # 영역과 해당 영역에 대한 Ranked Precision 점수 출력
        print(f"\nArea: {area} | Ranked Precision: {ranked_precision_value}")
        
        for rank, (_, row) in enumerate(group.iterrows(), 1):
            print(f"  Rank {rank}: knowledgeTag {row['knowledgeTag']} | Chapter: {row['후속학습_Chapter_Name']} | 계열화: {row['계열화']} | 영역: {row['영역']}")


# 로깅 설정
def log():
    # 1. logger instance 설정
    logger = logging.getLogger(__name__)

    # 2. formatter 생성 (로그 출력/저장에 사용할 날짜 + 로그 메시지)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 3. handler 생성 (streamHandler : 콘솔 출력용 // fileHandler : 파일 기록용)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler("server.log")    # 로그를 기록할 파일 이름 지정

    # 4. logger instance에 formatter 설정 (각각의 Handler에 formatter 설정 적용)
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # 5. logger instance에 handler 추가 (입력받는 log에 handler사용)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    # 6. 기록할 log level 지정하기
    logger.setLevel(level=logging.DEBUG)    # INFO 레벨로 지정하면, INFO 레벨보다 낮은 DEBUG 로그는 무시함.
                                            # Python의 기본 logging 시스템의 레벨은 WARNING으로 설정되어 있음.
                                            # 따라서 특별한 설정을 하지 않으면, WARNING 레벨 이상만 기록됨.

    # 설정된 log setting 반환
    return logger

# log 함수 호출
logger = log()

# 프로그램 시작 시간 기록
logger.debug("==========PROGRAM START==========")

def main():
    start_time = time.time()  # 전체 실행 시간 측정 시작
    logger.info("프로그램 실행 시작")

    # 데이터 로딩 진행 상황을 업데이트하기 위해 total_steps 설정
    total_steps = 100
    
    with tqdm(total=total_steps, desc='데이터 로딩 진행 상황', ncols=100) as pbar:
        # MongoDB에서 데이터 가져오기
        logger.info("MongoDB에서 데이터 로드 시작")
        mongo_start_time = time.time()
        merged_df = load_qa_irt_theta_data()  # MongoDB 데이터를 로드하여 merged_df 생성
        mongo_end_time = time.time()
        mongo_duration = mongo_end_time - mongo_start_time
        pbar.update(total_steps // 3)  # 33% 완료 (1/3)
        logger.info(f"MongoDB 데이터 로드 완료: {mongo_duration:.2f}초")

        # PostgreSQL에서 데이터 가져오기
        logger.info("PostgreSQL에서 education_data 로드 시작")
        postgres_start_time = time.time()
        education_data = load_education_data()  # PostgreSQL에서 education_2022 데이터를 로드
        postgres_end_time = time.time()
        postgres_duration = postgres_end_time - postgres_start_time
        pbar.update(total_steps // 3)  # 66% 완료 (2/3)
        logger.info(f"PostgreSQL education_data 로드 완료: {postgres_duration:.2f}초")

        # label_math 데이터 가져오기
        logger.info("PostgreSQL에서 label_math_ele 로드 시작")
        label_math_start_time = time.time()
        label_math_ele = load_label_math_ele()  # PostgreSQL에서 label_math_ele 데이터를 로드
        label_math_end_time = time.time()
        label_math_duration = label_math_end_time - label_math_start_time
        pbar.update(total_steps - (total_steps // 3 * 2))  # 100% 완료 (3/3)
        logger.info(f"PostgreSQL label_math_ele 로드 완료: {label_math_duration:.2f}초")
    
    end_time = time.time()
    total_time = end_time - start_time

    # 데이터 로딩 시간 출력
    logger.info(f"merged_df 데이터 로딩 시간: {mongo_duration:.2f}초")
    logger.info(f"education_data 데이터 로딩 시간: {postgres_duration:.2f}초")
    logger.info(f"label_math_ele 데이터 로딩 시간: {label_math_duration:.2f}초")
    logger.info(f"데이터 로딩 완료. 총 소요 시간: {total_time:.2f}초")

    # 데이터 처리 및 병합
    logger.info("데이터 처리 및 병합 시작")
    merged_df_with_education = process_merged_data()  # merged_df와 education_data 데이터를 병합한 DataFrame
    logger.info("데이터 처리 및 병합 완료")

    # 모델 로드
    logger.info("모델 로드 시작")
    model = load_model('enhanced_binary_nn_model.pth')  # 모델 로드 함수 호출
    logger.info("모델 로드 완료")

    # 학습자 ID 입력
    learner_id = None
    while True:
        learner_id = input("학습자 ID를 입력하세요: ").strip()
        if learner_id in merged_df_with_education['learnerID'].values:  # 병합된 데이터를 기준으로 학습자 ID 확인
            logger.info(f"학습자 ID '{learner_id}' 확인됨")
            break
        else:
            logger.warning(f"유효하지 않은 학습자 ID '{learner_id}' 입력됨. 다시 시도 필요.")

    # 추천 데이터 생성
    logger.info("추천 데이터 생성 시작")
    unique_id_name_pairs, unique_id_semesters, predecessors, successors, education_mapping = process_label_math_data(label_math_ele, education_data)

    # 추천 데이터 생성
    prerequisite_recommendations_df, successor_recommendations_df = recommend_all(
        learner_id, merged_df_with_education, model, unique_id_name_pairs, unique_id_semesters, education_mapping, predecessors, successors
    )

    # 추천할 내용이 있는지 확인
    if prerequisite_recommendations_df.empty and successor_recommendations_df.empty:
        logger.info("추천할 이전 학습 개념이 없습니다.")
        print("추천할 이전 학습 개념이 없습니다.")
    else:
        # 추천할 항목이 있으면 출력
        logger.info(f"추천 데이터 생성 완료, 학습자 {learner_id}에 대한 추천 결과 출력")
        top_prerequisite_recommendations_df = recommend_one_per_area(prerequisite_recommendations_df)
        top_successor_recommendations_df = recommend_one_per_area_successors(successor_recommendations_df, merged_df_with_education)

        print(f"Top recommended prerequisites for learner {learner_id}:")
        print(top_prerequisite_recommendations_df)

        print(f"Top recommended successors for learner {learner_id}:")
        print(top_successor_recommendations_df)

        # 실제 관련된 후속 학습 개념을 successor_recommendations_df로 설정
        relevant_items = set(successor_recommendations_df['knowledgeTag'])  # 실제 관련된 학습 개념 (올바른 선수 학습 개념)

        # 영역별로 랭크 및 Ranked Precision 출력
        print(f"\nTop recommended successors for learner {learner_id} (ranked by area with Ranked Precision):")
        rank_by_area(top_successor_recommendations_df, relevant_items)


        # 데이터베이스 적재 여부 선택
        while True:
            choice = input("추천 데이터를 데이터베이스에 적재하려면 1번, 적재하지 않으려면 2번을 입력하세요: ").strip()
            if choice == '1':
                logger.info("추천 데이터를 데이터베이스에 적재합니다.")
                recommend_concepts(learner_id, top_prerequisite_recommendations_df, top_successor_recommendations_df)
                logger.info("추천 데이터가 데이터베이스에 적재되었습니다.")
                print("추천 데이터가 데이터베이스에 적재되었습니다.")
                break
            elif choice == '2':
                logger.info("추천 데이터를 데이터베이스에 적재하지 않기로 선택했습니다.")
                print("추천 데이터가 적재되지 않았습니다.")
                break
            else:
                logger.warning("잘못된 입력이 발생했습니다. 1번 또는 2번을 입력하세요.")

if __name__ == "__main__":
    main()

# 프로그램 종료 시간 기록
logger.debug("==========PROGRAM FINISH==========")