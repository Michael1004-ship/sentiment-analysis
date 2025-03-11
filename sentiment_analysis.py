# 필요한 라이브러리 임포트
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
import os
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import language_v1
from transformers import pipeline
import csv
from sentistrength import PySentiStr
import nltk
from nltk.corpus import wordnet
import feedparser
import time
import re
from random import randint
import seaborn as sns  # 추가

# Google API 인증 파일 경로 (실제 경로로 업데이트 필요)
GOOGLE_APPLICATION_CREDENTIALS = "C:\\Users\\user\\Desktop\\연구\\국가별 및 신문사별 감정 강도 비교 연구\\코딩\\comparative-sentiment-analysis-c0b363950560.json"
HUGGINGFACE_API_KEY = "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"

# 필요한 NLTK 데이터 다운로드
try:
    nltk.download("wordnet", quiet=True)
except:
    print("NLTK wordnet 다운로드 실패 - 오프라인이거나 NLTK 설치에 문제가 있습니다.")

# 뉴스 소스 정의
NEWS_SOURCES = {
    "USA": {
        "The New York Times": {"bias": "liberal"},
        "CNN": {"bias": "liberal"},
        "Fox News": {"bias": "conservative"}
    },
    "UK": {
        "The Guardian": {"bias": "liberal"},
        "The Telegraph": {"bias": "conservative"},
        "BBC News": {"bias": "neutral"}
    }
}

# 결과 저장 경로 지정
RESULTS_DIR = "C:\\Users\\user\\Desktop\\연구\\국가별 및 신문사별 감정 강도 비교 연구\\코딩\\결과"

# 디버그 모드 설정
DEBUG_MODE = True

def debug_print(message, data=None, important=False):
    """디버그 메시지 출력 함수"""
    if DEBUG_MODE:
        if important:
            print("\n" + "="*50)
            print(f"🔍 {message}")
            print("="*50)
        else:
            print(f"🔹 {message}")
        
        if data is not None:
            if isinstance(data, str) and len(data) > 300:
                print(f"{data[:300]}... (생략됨)")
            else:
                print(data)

# 오류 메시지 설정
def setup_error_handling():
    """오류 처리 설정"""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    print("알림: 국가별, 신문사별 감정 강도 비교를 위한 분석을 시작합니다.")

# WordNet을 활용한 동의어 확장
def get_synonyms(keyword):
    """단어의 동의어 목록 반환"""
    synonyms = set()
    try:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
    except Exception as e:
        debug_print(f"동의어 찾기 오류: {e}")
    return list(synonyms)

# 소스 도메인 가져오기
def get_source_domain(source):
    """뉴스 소스의 도메인 반환"""
    source_domains = {
        'CNN': 'cnn.com',
        'Fox News': 'foxnews.com',
        'The Guardian': 'theguardian.com',
        'The New York Times': 'nytimes.com',
        'BBC News': 'bbc.com,bbc.co.uk',
        'The Telegraph': 'telegraph.co.uk',
        'Reuters': 'reuters.com',
        'CNBC': 'cnbc.com',
        'Bloomberg': 'bloomberg.com'
    }
    
    return source_domains.get(source, '')

# 소스 일치 여부 확인
def is_same_source(found_source, target_source):
    """발견된 소스가 목표 소스와 일치하는지 확인"""
    found_lower = found_source.lower()
    target_lower = target_source.lower()
    
    # 정확히 일치
    if found_lower == target_lower:
        return True
    
    # 부분 문자열 확인
    if target_lower in found_lower or found_lower in target_lower:
        return True
    
    # CNN, NYT 등 약어 확인
    abbreviations = {
        'cnn': ['cnn'],
        'fox news': ['fox', 'fox news'],
        'the guardian': ['guardian', 'the guardian'],
        'the new york times': ['nyt', 'ny times', 'new york times'],
        'bbc news': ['bbc'],
        'the telegraph': ['telegraph'],
        'reuters': ['reuters']
    }
    
    target_abbr = abbreviations.get(target_lower, [])
    for abbr in target_abbr:
        if abbr in found_lower:
            return True
    
    return False

# Google News RSS에서 뉴스 검색
def get_google_news_rss(keyword, source=None):
    """Google News RSS에서 키워드로 뉴스 검색"""
    # 소스 지정이 있으면 소스 필터링 추가
    query = keyword
    if source:
        query = f"{keyword} site:{get_source_domain(source)}"
    
    # URL 인코딩
    query = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    debug_print(f"Google News RSS 요청 URL: {url}")
    
    try:
        feed = feedparser.parse(url)
        
        articles = []
        for entry in feed.entries[:15]:  # 조금 더 많은 결과를 가져와서 필터링할 여지 확보
            # 소스 추출
            title_parts = entry.title.split(" - ")
            entry_source = title_parts[-1].strip() if len(title_parts) > 1 else "Unknown"
            
            # 소스 확인 - 더 엄격한 검사
            if source and not is_same_source(entry_source, source):
                debug_print(f"소스 불일치: '{entry_source}' ≠ '{source}' - 건너뜀")
                continue
                
            article = {
                'title': title_parts[0].strip(),
                'content': entry.description if hasattr(entry, 'description') else title_parts[0].strip(),
                'source': entry_source,
                'url': entry.link,
                'published_at': entry.published if hasattr(entry, 'published') else datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            articles.append(article)
        
        debug_print(f"'{keyword}' 검색 결과: {len(articles)}개 기사 발견")
        return articles
        
    except Exception as e:
        debug_print(f"Google News RSS 요청 중 오류 발생: {str(e)}")
        return []

# 키워드 확장 + Google News 검색 결합
def search_expanded_news(keyword, source=None):
    """키워드 확장 후 뉴스 검색"""
    debug_print(f"'{keyword}' 확장 검색 시작 (소스: {source})", important=True)
    
    # Step 1: 원래 키워드로 Google News 검색
    initial_news = get_google_news_rss(keyword, source)
    
    if len(initial_news) >= 5:  # 충분한 결과가 있으면 바로 반환
        debug_print(f"원래 키워드로 충분한 결과({len(initial_news)}개)를 찾았습니다.")
        return initial_news[:5]  # 상위 5개만 반환
    
    # Step 2: 키워드 확장이 필요한 경우
    debug_print(f"결과가 부족하여 키워드 확장을 시도합니다.")
    expanded_keywords = [keyword]  # 원래 키워드 포함
    
    # 단일 단어인 경우 WordNet 동의어 추가
    if ' ' not in keyword and len(keyword) > 3:
        synonyms = get_synonyms(keyword)[:3]  # 상위 3개 동의어만 사용
        expanded_keywords.extend(synonyms)
    
    # 중복 제거
    expanded_keywords = list(set(expanded_keywords))
    debug_print(f"확장된 키워드: {expanded_keywords}")
    
    # Step 3: 확장된 키워드로 추가 검색
    all_news = initial_news.copy()  # 초기 결과 포함
    
    for k in expanded_keywords:
        if k == keyword:  # 원래 키워드는 이미 검색했으므로 건너뜀
            continue
            
        debug_print(f"확장 키워드 '{k}'로 검색 중...")
        additional_news = get_google_news_rss(k, source)
        
        # 중복 제거하며 추가
        for article in additional_news:
            # URL로 중복 확인
            if not any(existing['url'] == article['url'] for existing in all_news):
                all_news.append(article)
    
    debug_print(f"최종 검색 결과: {len(all_news)}개 기사")
    return all_news[:5]  # 최대 5개만 반환

# 감정 분석 함수들

def get_vader_sentiment(text):
    """VADER를 사용하여 감정 점수 분석"""
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        # compound 점수는 -1에서 1 사이의 값
        compound_score = sentiment_scores['compound']
        print(f"VADER 원본 점수: {compound_score}")
        return compound_score
    except Exception as e:
        print(f"VADER 감정 분석 오류: {e}")
        return 0

def get_sentistrength_sentiment(text):
    """SentiStrength를 사용하여 감정 점수 분석"""
    try:
        # SentiStrength 초기화
        senti = PySentiStr()
        
        # SentiStrength.jar 파일 경로 설정
        senti.setSentiStrengthPath("C:\\Users\\user\\Desktop\\연구\\국가별 및 신문사별 감정 강도 비교 연구\\코딩\\SentiStrength.jar")
        
        # 언어 폴더 경로 설정
        senti.setSentiStrengthLanguageFolderPath("C:\\Users\\user\\Desktop\\연구\\국가별 및 신문사별 감정 강도 비교 연구\\코딩\\SentiStrengthDataEnglishOctober2019")
        
        # dual 점수로 분석 (긍정 및 부정 점수 각각 반환)
        result = senti.getSentiment(text, score='dual')
        
        # 결과는 튜플 형태: (positive_score, negative_score)
        positive_score, negative_score = result[0]
        
        print(f"SentiStrength 원본 점수: 긍정={positive_score}, 부정={negative_score}")
        return positive_score, negative_score
    except Exception as e:
        print(f"SentiStrength 감정 분석 오류: {e}")
        # 오류 발생 시 피드백 추가
        print(f"경로 확인: JAR={senti.SentiStrengthPath}, 데이터={senti.SentiStrengthLanguageFolderPath}")
        print("Java가 설치되어 있는지 확인하세요: 'java -version' 명령어로 확인 가능")
        return 1, -1  # 중립 값 반환

def get_google_sentiment(text):
    """Google Natural Language API를 사용하여 감정 점수 분석"""
    try:
        # Google Cloud 인증 설정
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
        
        # 클라이언트 초기화
        client = language_v1.LanguageServiceClient()
        
        # 문서 객체 생성
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT
        )
        
        # 감정 분석 수행
        sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
        
        # Google API는 -1.0 ~ 1.0 범위의 점수를 반환
        score = sentiment.score
        print(f"Google API 원본 점수: {score}")
        return score
    except Exception as e:
        print(f"Google 감정 분석 오류: {e}")
        return 0  # 오류 발생 시 중립값 반환

def get_huggingface_sentiment(text):
    try:
        # 모델 명시적 지정으로 경고 제거
        sentiment_analyzer = pipeline("sentiment-analysis", 
                                    model="distilbert-base-uncased-finetuned-sst-2-english")
        # 텍스트 길이 제한 (너무 긴 텍스트는 잘라서 분석)
        max_length = 512
        truncated_text = text[:max_length] if len(text) > max_length else text
        result = sentiment_analyzer(truncated_text)[0]
        
        # NEGATIVE 결과에 대한 처리 개선
        if result['label'] == 'POSITIVE':
            score = result['score']
        else:
            score = 1 - result['score']  # 더 직관적인 변환
            
        print(f"Hugging Face 점수: {score} (원래 레이블: {result['label']})")
        return score
    except Exception as e:
        print(f"Hugging Face 감정 분석 오류: {e}")
        return 0.5

# 점수 정규화 함수들

def normalize_vader_score(score):
    """VADER 점수를 0~1 범위로 정규화"""
    normalized_score = (score + 1) / 2
    return normalized_score

def normalize_sentistrength_score(positive_score, negative_score):
    """SentiStrength 점수를 0~1 범위로 정규화"""
    combined_score = (positive_score + (6 + negative_score)) / 10
    return combined_score

def normalize_google_score(score):
    """Google API 점수를 0~1 범위로 정규화"""
    normalized_score = (score + 1) / 2
    return normalized_score

# 최종 감정 점수 계산 함수

def calculate_final_sentiment(vader_score, sentistrength_positive, sentistrength_negative, google_score, huggingface_score):
    """각 도구의 정규화된 점수를 가중 평균하여 최종 감정 점수 계산"""
    # 각 점수 정규화
    normalized_vader = normalize_vader_score(vader_score)
    normalized_sentistrength = normalize_sentistrength_score(sentistrength_positive, sentistrength_negative)
    normalized_google = normalize_google_score(google_score)
    
    # 정규화된 점수 출력
    print(f"정규화된 VADER 점수: {normalized_vader:.4f}")
    print(f"정규화된 SentiStrength 점수: {normalized_sentistrength:.4f}")
    print(f"정규화된 Google API 점수: {normalized_google:.4f}")
    print(f"정규화된 Hugging Face 점수: {huggingface_score:.4f}")
    
    # 극단적인 점수가 있을 경우 가중치 조정
    # Hugging Face가 0.1 미만이거나 0.9 초과일 경우 가중치 낮춤
    if huggingface_score < 0.1 or huggingface_score > 0.9:
        vader_weight = 0.3
        sentistrength_weight = 0.4
        google_weight = 0.3
        huggingface_weight = 0.0  # 가중치 제거
        print("Hugging Face 점수가 극단적이어서 계산에서 제외합니다.")
    else:
        vader_weight = 0.25
        sentistrength_weight = 0.3
        google_weight = 0.25
        huggingface_weight = 0.2
        
    # 가중 평균 계산 (Hugging Face 제외 가능성 있음)
    if huggingface_weight > 0:
        final_score = (vader_weight * normalized_vader + 
                     sentistrength_weight * normalized_sentistrength + 
                     google_weight * normalized_google + 
                     huggingface_weight * huggingface_score)
    else:
        # 합계가 1이 되도록 정규화
        total_weight = vader_weight + sentistrength_weight + google_weight
        final_score = (vader_weight * normalized_vader + 
                     sentistrength_weight * normalized_sentistrength + 
                     google_weight * normalized_google) / total_weight
    
    return final_score

# 기사 감정 분석 함수

def analyze_article_sentiment(article, country, search_keyword=None):
    """기사 감정 분석 수행"""
    title = article['title']
    content = article['content']
    source = article['source']
    text = f"{title}. {content}"  # 제목과 내용을 합쳐서 분석
    
    print(f"\n입력 뉴스 기사: '{title}'")
    print(f"출처: {source} (국가: {country})")
    print(f"내용 일부: {content[:100]}...\n")
    
    # 각 도구를 사용하여 감정 점수 분석
    vader_score = get_vader_sentiment(text)
    sentistrength_pos, sentistrength_neg = get_sentistrength_sentiment(text)
    google_score = get_google_sentiment(text)
    huggingface_score = get_huggingface_sentiment(text)
    
    # 최종 감정 점수 계산
    final_score = calculate_final_sentiment(
        vader_score, 
        sentistrength_pos, 
        sentistrength_neg, 
        google_score, 
        huggingface_score
    )
    
    # 결과 출력
    print("\n===== 감정 분석 결과 =====")
    print(f"VADER 감정 점수: {vader_score} (정규화: {normalize_vader_score(vader_score):.4f})")
    print(f"SentiStrength 점수: 긍정={sentistrength_pos}, 부정={sentistrength_neg} (정규화: {normalize_sentistrength_score(sentistrength_pos, sentistrength_neg):.4f})")
    print(f"Google API 감정 점수: {google_score} (정규화: {normalize_google_score(google_score):.4f})")
    print(f"Hugging Face 감정 점수: {huggingface_score:.4f}")
    print(f"최종 감정 점수: {final_score:.4f}")
    
    # 감정 해석
    if final_score > 0.65:
        sentiment = "매우 긍정적"
    elif final_score > 0.55:
        sentiment = "긍정적"
    elif final_score > 0.45:
        sentiment = "중립적"
    elif final_score > 0.35:
        sentiment = "부정적"
    else:
        sentiment = "매우 부정적"
    
    print(f"감정 해석: {sentiment}")
    
    # 결과 반환
    return {
        'title': title,
        'source': source,
        'country': country,
        'bias': get_source_bias(source),
        'search_keyword': search_keyword,
        'final_score': final_score,
        'vader_score': normalize_vader_score(vader_score),
        'sentistrength_score': normalize_sentistrength_score(sentistrength_pos, sentistrength_neg),
        'google_score': normalize_google_score(google_score),
        'huggingface_score': huggingface_score,
        'sentiment': sentiment,
        'url': article.get('url', '')
    }

def get_source_bias(source):
    """신문사의 정치적 성향 반환"""
    for country, sources in NEWS_SOURCES.items():
        if source in sources:
            return sources[source]['bias']
    return "unknown"

# 국가별, 신문사별 감정 강도 비교 함수

def compare_news_sources(keywords, articles_per_source=3):
    """여러 뉴스 소스에서 키워드별 감정 분석 비교"""
    results = []
    
    # 원하는 뉴스 소스로 정확히 제한
    sources = {
        'CNN': 'US',
        'Fox News': 'US',
        'The Guardian': 'UK',
        'The New York Times': 'US',
        'The Telegraph': 'UK',
        'BBC News': 'UK'
    }
    
    for keyword in keywords:
        debug_print(f"\n키워드 '{keyword}'에 대한 분석 시작...", important=True)
        
        for source, country in sources.items():
            debug_print(f"{source}({country})에서 '{keyword}' 검색 중...")
            
            # Google News RSS로 기사 검색
            articles = search_expanded_news(keyword, source)
            
            if not articles:
                debug_print(f"❌ {source}에서 '{keyword}'에 대한 기사를 찾을 수 없습니다.")
                continue
            
            # 찾은 기사 개수 제한
            articles = articles[:articles_per_source]
            
            # 각 기사 분석
            for article in articles:
                # 중복 API 호출 방지를 위한 지연
                time.sleep(randint(1, 3))
                
                result = analyze_article_sentiment(article, country, search_keyword=keyword)
                
                if result:
                    results.append(result)
                    debug_print(f"✅ 분석 완료: {result['title']} (점수: {result['final_score']})")
    
    return results

# 결과 시각화 및 저장 함수

def get_date_folder():
    """오늘 날짜를 기준으로 폴더명 생성 (예: 20240601)"""
    today = datetime.datetime.now()
    return today.strftime("%Y%m%d")

def get_timestamp():
    """현재 시간을 파일명에 적합한 형식으로 반환 (예: 20240601_143045)"""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def get_result_folder():
    """날짜별 결과 폴더 경로 반환 및 생성"""
    date_folder = get_date_folder()
    result_path = os.path.join(RESULTS_DIR, date_folder)
    
    # 결과 폴더가 없으면 생성
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # 날짜별 폴더가 없으면 생성
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"새 날짜 폴더 생성: {result_path}")
    
    return result_path

def sample_article_review(df, sample_size=5):
    """Sample article review for sentiment analysis validation"""
    print("\n🔍 Sample Article Sentiment Analysis Review")
    if len(df) < sample_size:
        sample_size = len(df)
        print(f"⚠️ Only {sample_size} samples available for review.")
    
    sample_articles = df.sample(sample_size)
    for idx, row in sample_articles.iterrows():
        print(f"\nArticle Title: {row['title']}")
        print(f"Source: {row['source']} (Country: {row['country']})")
        print(f"Search Keyword: {row.get('search_keyword', 'No info')}")
        print(f"Final Sentiment Score: {row['final_score']:.4f} ({row['sentiment']})")
        print(f"Tool Scores: VADER={row['vader_score']:.2f}, SentiStrength={row['sentistrength_score']:.2f}, " +
              f"Google={row['google_score']:.2f}, HuggingFace={row['huggingface_score']:.2f}")
        print(f"URL: {row.get('url', 'No info')}")
        print("=" * 50)

def visualize_sentiment_by_source(df, result_folder, timestamp):
    """Compare sentiment score distribution by news source"""
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='source', y='final_score', data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title('Sentiment Score Distribution by News Source')
    plt.ylabel('Sentiment Score (0-1)')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'source_sentiment_boxplot_{timestamp}.png'))
    print(f"📊 News source sentiment distribution graph saved.")

def compare_sentiment_tools(df, result_folder, timestamp):
    """Compare sentiment scores across different analysis tools"""
    plt.figure(figsize=(10, 6))
    tools = ['vader_score', 'sentistrength_score', 'google_score', 'huggingface_score']
    tool_labels = ['VADER', 'SentiStrength', 'Google NLP', 'HuggingFace']
    
    # Box plot
    ax = sns.boxplot(data=df[tools])
    ax.set_xticklabels(tool_labels)
    plt.title('Sentiment Score Comparison Across Analysis Tools')
    plt.ylabel('Sentiment Score (0-1)')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'tool_comparison_{timestamp}.png'))
    
    # Correlation analysis
    plt.figure(figsize=(10, 8))
    correlation = df[tools].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=tool_labels, yticklabels=tool_labels)
    plt.title('Correlation Between Sentiment Analysis Tools')
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'tool_correlation_{timestamp}.png'))
    print(f"📊 Tool comparison graphs saved.")

def analyze_bias_influence(df, result_folder, timestamp):
    """Analyze the influence of political bias on sentiment analysis"""
    plt.figure(figsize=(12, 6))
    
    # Political bias sentiment distribution
    sns.boxplot(x='bias', y='final_score', data=df)
    plt.title('Sentiment Score Distribution by Political Bias')
    plt.xlabel('Political Bias')
    plt.ylabel('Sentiment Score (0-1)')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'bias_influence_{timestamp}.png'))
    
    # Tool-specific bias analysis
    tools = ['vader_score', 'sentistrength_score', 'google_score', 'huggingface_score']
    tool_labels = ['VADER', 'SentiStrength', 'Google NLP', 'HuggingFace']
    
    plt.figure(figsize=(15, 10))
    for i, tool in enumerate(tools):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='bias', y=tool, data=df)
        plt.title(f'{tool_labels[i]}')
        plt.xlabel('Political Bias')
        plt.ylabel('Sentiment Score (0-1)')
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'bias_by_tool_{timestamp}.png'))
    print(f"📊 Political bias influence graphs saved.")

def evaluate_sentiment_reliability(df, result_folder, timestamp):
    """Comprehensive evaluation of sentiment analysis reliability"""
    print("\n=== Starting Sentiment Analysis Reliability Evaluation ===")
    
    # 1. Sample article review
    sample_article_review(df)
    
    # 2. News source sentiment comparison
    visualize_sentiment_by_source(df, result_folder, timestamp)
    
    # 3. Analysis tool comparison
    compare_sentiment_tools(df, result_folder, timestamp)
    
    # 4. Political bias influence analysis
    analyze_bias_influence(df, result_folder, timestamp)
    
    print("\n=== Sentiment Analysis Reliability Evaluation Complete ===")
    print(f"All evaluation graphs have been saved to {result_folder} folder.")

def visualize_results(results):
    """Visualize analysis results and evaluate reliability"""
    if not results:
        print("No results to visualize.")
        return
    
    # Get date-based result folder
    result_folder = get_result_folder()
    
    # Get current timestamp
    timestamp = get_timestamp()
    
    # Create dataframe
    df = pd.DataFrame(results)
    
    # Add analysis time to title
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    title_suffix = f" (Analysis: {current_time})"
    
    # 1. Country-based sentiment comparison
    plt.figure(figsize=(10, 6))
    country_avg = df.groupby('country')['final_score'].mean().sort_values(ascending=False)
    country_avg.plot(kind='bar', color='skyblue')
    plt.title('Average Sentiment Score by Country' + title_suffix)
    plt.ylabel('Sentiment Score (0-1)')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)  # Neutral line
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'country_sentiment_{timestamp}.png'))
    
    # 2. News source sentiment comparison
    plt.figure(figsize=(12, 6))
    source_avg = df.groupby('source')['final_score'].mean().sort_values(ascending=False)
    source_avg.plot(kind='bar', color='lightgreen')
    plt.title('Average Sentiment Score by News Source' + title_suffix)
    plt.ylabel('Sentiment Score (0-1)')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)  # Neutral line
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'source_sentiment_{timestamp}.png'))
    
    # 3. Political bias sentiment comparison
    plt.figure(figsize=(8, 6))
    bias_avg = df.groupby('bias')['final_score'].mean().sort_values(ascending=False)
    colors = {'liberal': 'blue', 'conservative': 'red', 'neutral': 'gray', 'unknown': 'black'}
    bias_avg.plot(kind='bar', color=[colors.get(x, 'black') for x in bias_avg.index])
    plt.title('Average Sentiment Score by Political Bias' + title_suffix)
    plt.ylabel('Sentiment Score (0-1)')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)  # Neutral line
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'bias_sentiment_{timestamp}.png'))
    
    # 4. Analysis tool comparison
    plt.figure(figsize=(10, 6))
    tools = ['vader_score', 'sentistrength_score', 'google_score', 'huggingface_score']
    tool_avg = df[tools].mean()
    tool_avg.plot(kind='bar', color='orange')
    plt.title('Average Sentiment Score by Analysis Tool' + title_suffix)
    plt.ylabel('Sentiment Score (0-1)')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)  # Neutral line
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'tool_sentiment_{timestamp}.png'))
    
    print(f"\nVisualization images have been saved to {result_folder} folder.")
    
    # Run reliability evaluation
    evaluate_sentiment_reliability(df, result_folder, timestamp)

def save_results_to_csv(results, filename=None):
    """분석 결과를 CSV 파일로 저장"""
    if not results:
        print("저장할 결과가 없습니다.")
        return
    
    # 날짜별 결과 폴더 가져오기
    result_folder = get_result_folder()
    
    # 파일명에 시간 추가
    timestamp = get_timestamp()
    if filename is None:
        filename = f"news_sentiment_analysis_{timestamp}.csv"
    
    # 결과에 분석 시간 추가
    analysis_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        r['analysis_time'] = analysis_time
    
    # 전체 파일 경로
    filepath = os.path.join(result_folder, filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['search_keyword', 'title', 'source', 'country', 'bias', 'final_score', 
                     'vader_score', 'sentistrength_score', 'google_score', 'huggingface_score',
                     'sentiment', 'url', 'analysis_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n분석 결과가 {filepath}에 저장되었습니다.")

# 메인 함수
def main():
    """메인 실행 함수"""
    print("알림: 국가별, 신문사별 감정 강도 비교를 위한 분석을 시작합니다.")
    
    # 검색할 키워드 목록
    keywords = [
        "W.T.O", "WTO", "World Trade Organization",
        "tariff", "tariffs",
        "trade war",
        "free trading", "free trade",
        "trade liberalization",
        "trade agreements", "trade deal",
        "global trade", "international trade",
        "import export regulations",
        "trade deficit", "trade surplus",
        "protectionism", "trade barriers"
    ]
    
    # 테스트 단계에서는 키워드 수 제한
    if DEBUG_MODE:
        test_keywords = ["trade"]
        debug_print(f"테스트 모드: {test_keywords} 키워드로 테스트 실행", important=True)
        results = compare_news_sources(test_keywords, articles_per_source=1)
        
        if not results:
            debug_print("❌ 테스트에서 결과가 나오지 않았습니다.")
            return
        else:
            debug_print(f"✅ 테스트 성공: {len(results)}개 결과 생성")
            
            # 테스트 후 계속 진행할지 물어봄
            continue_analysis = input("테스트 성공! 전체 분석을 진행하시겠습니까? (y/n): ").lower() == 'y'
            if not continue_analysis:
                return
    
    # 전체 키워드로 실행
    results = compare_news_sources(keywords, articles_per_source=3)
    
    # 결과 저장 및 시각화
    if results:
        save_results_to_csv(results)
        visualize_results(results)
        debug_print(f"✅ 총 {len(results)}개 기사 분석 완료. 결과 저장됨.")
    else:
        debug_print("❌ 분석 결과가 없습니다.")

if __name__ == "__main__":
    main()