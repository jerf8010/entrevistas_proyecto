import streamlit as st
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import time
import queue
import sounddevice as sd
import wave
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import pandas as pd
import av

import spacy
from pyresparser import ResumeParser as RealResumeParser
import speech_recognition as sr
from pydub import AudioSegment
import cv2
from deepface import DeepFace

# Cargar modelo de lenguaje para NLP
nlp = spacy.load("es_core_news_md")

import re
import pdfplumber
from collections import defaultdict

# Lista de stopwords en español e inglés manuales
SPANISH_STOPWORDS = {
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 
    'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'es'
}

ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
    'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself'
}

class ResumeParser:
    def __init__(self):
        self.skill_keywords = {
            'tech': ['python', 'java', 'sql', 'machine learning', 'docker', 'aws'],
            'soft': ['comunicación', 'trabajo en equipo', 'liderazgo', 'adaptabilidad']
        }
        self.education_keywords = [
            'universidad', 'grado', 'licenciatura', 'maestría', 
            'doctorado', 'diplomado', 'bachillerato'
        ]
    
    def clean_text(self, text):
        """Limpia el texto y elimina stopwords"""
        words = re.findall(r'\b\w+\b', text.lower())
        return ' '.join([w for w in words if w not in SPANISH_STOPWORDS and w not in ENGLISH_STOPWORDS])
    
    def extract_text(self, filepath):
        """Extrae texto de PDFs"""
        try:
            with pdfplumber.open(filepath) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return ""
    
    def analyze_resume(self, filepath):
        raw_text = self.extract_text(filepath)
        clean_text = self.clean_text(raw_text)
        
        # Extraer habilidades
        found_skills = []
        for category, skills in self.skill_keywords.items():
            for skill in skills:
                if re.search(rf'\b{re.escape(skill)}\b', clean_text):
                    found_skills.append(skill.title())
        
        # Extraer experiencia
        experience = self.extract_experience(raw_text)
        
        # Extraer educación
        education = self.extract_education(raw_text)
        
        return {
            'skills': list(set(found_skills)),
            'experience': experience,
            'education': education,
            'raw_text': raw_text[:1000] + " [...]" if len(raw_text) > 1000 else raw_text
        }
    
    def extract_experience(self, text):
        """Extrae años de experiencia"""
        patterns = [
            r'(\d+)\s*(años|año)\s+(de\s+)?experiencia',
            r'experience\s*:\s*(\d+)\s*(years|year)',
            r'(\d+)\+?\s*(years?|años?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0
    
    def extract_education(self, text):
        """Identifica formación académica"""
        education = []
        text_lower = text.lower()
        
        for edu in self.education_keywords:
            if re.search(rf'\b{re.escape(edu)}\b', text_lower):
                education.append(edu.capitalize())
        
        return education if education else ["Educación no especificada"]

class VoiceAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def extract_features(self, audio_path):
        try:
            # Convertir a formato WAV si es necesario
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path.replace('.mp3', '.wav')
            audio.export(wav_path, format="wav")
            
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                
            return {
                'duration': len(audio)/1000,  # segundos
                'sample_rate': audio.frame_rate,
                'channels': audio.channels
            }
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return {}
    
    def analyze_voice_characteristics(self, features):
        try:
            clarity = min(features.get('sample_rate', 0) / 44100, 1.0)
            tone = 'neutral'
            
            if features.get('duration', 0) < 2:
                tone = 'rapido'
            elif features.get('duration', 0) > 10:
                tone = 'lento'
                
            return {
                'tono': tone,
                'claridad': round(clarity, 2),
                'volumen': features.get('max_volume', 0.5)
            }
        except:
            return {'tono': 'neutral', 'claridad': 0.8}

class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def transcribe(self, audio_path):
        try:
            audio = AudioSegment.from_file(audio_path)
            wav_path = "/tmp/temp_audio.wav"
            audio.export(wav_path, format="wav")
            
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='es-ES')
                
            return {
                'text': text,
                'segments': [{'start': 0, 'end': len(audio)/1000, 'text': text}],
                'language': 'es'
            }
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return {'text': '', 'segments': [], 'language': 'es'}

class InterviewBot:
    def __init__(self):
        self.nlp = spacy.load("es_core_news_md")
    
    def generate_questions(self, skills, num_questions=5):
        questions = []
        question_templates = [
            "Describe tu experiencia con {skill}",
            "¿Qué proyectos has realizado usando {skill}?",
            "¿Cómo calificarías tu nivel en {skill} y por qué?",
            "Explica un desafío que hayas enfrentado con {skill}",
            "¿Cómo has aplicado {skill} en situaciones reales?"
        ]
        
        for skill in skills[:num_questions]:
            template = np.random.choice(question_templates)
            questions.append({
                'question': template.format(skill=skill),
                'skill': skill,
                'type': 'technical' if skill.lower() in ['python', 'java', 'sql'] else 'general'
            })
        
        return questions
    
    def evaluate_answer(self, question, answer, skills, position_reqs=None):
        doc = self.nlp(answer.lower())
        question_skill = question.get('skill', '').lower()
        
        # 1. Análisis de relevancia
        relevance = 1.0 if question_skill in answer.lower() else 0.0
        if not relevance:
            for token in doc:
                if token.text in question_skill or token.similarity(self.nlp(question_skill)[0]) > 0.7:
                    relevance = 0.7
                    break
        
        # 2. Análisis de profundidad
        depth = min(len(doc.ents) / 5, 1.0)  # Máx 5 entidades nombradas
        
        # 3. Coherencia (usando vectors de spaCy)
        question_vec = self.nlp(question['question']).vector
        answer_vec = doc.vector
        similarity = cosine_similarity([question_vec], [answer_vec])[0][0]
        
        # Puntaje compuesto
        score = min(10, (relevance * 4 + depth * 3 + similarity * 3) * 2.5)
        
        # Feedback detallado
        feedback = []
        if relevance < 0.5:
            feedback.append(f"La respuesta no menciona claramente {question_skill}.")
        if depth < 0.4:
            feedback.append("La respuesta carece de ejemplos concretos.")
        if similarity < 0.3:
            feedback.append("La respuesta parece desviarse de la pregunta.")
        
        return {
            'puntaje': round(score, 1),
            'feedback': " ".join(feedback) if feedback else "Buena respuesta, clara y relevante.",
            'metricas': {
                'relevancia': f"{relevance*100:.0f}%",
                'profundidad': f"{depth*100:.0f}%",
                'coherencia': f"{similarity*100:.0f}%"
            }
        }

class FacialEmotionAnalyzer:
    def analyze_frame(self, frame):
        try:
            # Convertir frame a formato que DeepFace entiende
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
            
            if isinstance(result, list):
                result = result[0]
                
            return {
                'dominant_emotion': result['dominant_emotion'],
                'emotions': result['emotion']
            }
        except Exception as e:
            print(f"Emotion analysis error: {str(e)}")
            return {'dominant_emotion': 'neutral', 'emotions': {}}
    
    def analyze_frames(self, frames):
        emotions = []
        for frame in frames:
            emotions.append(self.analyze_frame(frame))
        
        # Promedio de emociones
        avg_emotions = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0
        }
        
        for emo in emotions:
            for k, v in emo['emotions'].items():
                avg_emotions[k] += v / len(frames)
        
        dominant = max(avg_emotions.items(), key=lambda x: x[1])[0]
        
        return {
            'dominant_emotion': dominant,
            'emotions': avg_emotions
        }
# Clase principal para el análisis de contenido
class EnhancedContentMatcher:
    def __init__(self):
        self.position_requirements = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def set_position_requirements(self, requirements_text):
        self.position_requirements = requirements_text
        if requirements_text:
            self.vectorizer.fit([requirements_text])
    
    def vectorize(self, text):
        if not text or not hasattr(self.vectorizer, 'vocabulary_'):
            return None
        return self.vectorizer.transform([text])
    
    def cosine_similarity(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0
        return cosine_similarity(vec1, vec2)[0][0]
    
    def extract_skills(self, text):
        skills = []
        for part in text.split(','):
            for subpart in part.split(';'):
                for skill in subpart.split('.'):
                    skill = skill.strip().lower()
                    if skill and len(skill) > 2:
                        skills.append(skill)
        return list(set(skills))
    
    def analyze_position_match(self, skills):
        if not self.position_requirements:
            return {"error": "No position requirements set"}
        
        required_skills = self.extract_skills(self.position_requirements)
        candidate_skills = [s.lower() for s in skills]
        
        matching_skills = [skill for skill in required_skills if skill in candidate_skills]
        missing_skills = [skill for skill in required_skills if skill not in candidate_skills]
        
        match_percentage = len(matching_skills) / len(required_skills) * 100 if required_skills else 0
        
        return {
            "match_percentage": match_percentage,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "required_skills": required_skills,
            "candidate_skills": candidate_skills
        }
    
    def analyze_question_response_match(self, question, response_text):
        if not question or not response_text:
            return {"question_response_similarity": 0, "is_relevant": False}
        
        try:
            question_vec = self.vectorize(question)
            response_vec = self.vectorize(response_text)
            similarity = self.cosine_similarity(question_vec, response_vec) if (question_vec and response_vec) else 0
        except:
            similarity = 0
        
        return {
            "question_response_similarity": similarity,
            "is_relevant": similarity > 0.3
        }
    
    def analyze_content_match(self, skills, text):
        if not skills or not text:
            return {"match": 0, "matched_skills": []}
        
        text_lower = text.lower()
        matched_skills = [skill for skill in skills if skill.lower() in text_lower]
        
        return {
            "match": len(matched_skills) / len(skills) if skills else 0,
            "matched_skills": matched_skills
        }

# Clase para procesamiento de video
class LiveEmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.facial_analyzer = FacialEmotionAnalyzer()
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.frame_count % 5 == 0:
            try:
                emotion_result = self.facial_analyzer.analyze_frame(img)
                st.session_state.emotion_queue.put(emotion_result)
            except:
                st.session_state.emotion_queue.put({"dominant_emotion": "error"})
        
        return frame

# Funciones auxiliares
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def record_audio(duration, sample_rate=44100):
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return recording

def save_audio(recording, sample_rate=44100):
    try:
        temp_filename = f"temp_audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        audio_data = np.array(recording).flatten()
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        return temp_path
    except:
        return None

def analyze_response(audio_path, question, skills, facial_analyzer, video_frames=None):
    """
    Analiza una respuesta de entrevista de manera integral
    Args:
        audio_path: Ruta al archivo de audio de la respuesta
        question: Diccionario con la pregunta realizada
        skills: Lista de habilidades del candidato
        facial_analyzer: Instancia del analizador facial
        video_frames: Lista de frames de video (opcional)
    Returns:
        Dict con todos los análisis integrados
    """
    # 1. Transcripción de audio
    try:
        transcription_result = speech_to_text.transcribe(audio_path)
        transcription_text = transcription_result.get('text', '')
    except Exception as e:
        print(f"Error en transcripción: {str(e)}")
        transcription_text = ""

    # 2. Análisis de voz
    try:
        voice_features = voice_analyzer.extract_features(audio_path)
        voice_analysis = voice_analyzer.analyze_voice_characteristics(voice_features)
    except Exception as e:
        print(f"Error en análisis de voz: {str(e)}")
        voice_analysis = {'error': str(e)}

    # 3. Análisis facial (usando video frames si están disponibles)
    emotion_analysis = {'dominant_emotion': 'unknown', 'emotions': {}}
    try:
        if facial_analyzer:
            if video_frames and len(video_frames) > 0:
                emotion_analysis = facial_analyzer.analyze_frames(video_frames)
            else:
                # Si no hay frames, analizar un frame representativo
                emotion_analysis = facial_analyzer.analyze_frame(None)
    except Exception as e:
        print(f"Error en análisis facial: {str(e)}")

    # 4. Análisis de contenido
    content_analysis = content_matcher.analyze_content_match(
        skills, 
        transcription_text
    )

    # 5. Matching pregunta-respuesta
    question_response_match = content_matcher.analyze_question_response_match(
        question.get('question', ''), 
        transcription_text
    )

    # 6. Evaluación integral
    answer_evaluation = interview_bot.evaluate_answer(
        question,
        transcription_text,
        skills,
        content_matcher.position_requirements
    )

    # Resultado consolidado
    return {
        'transcription': {
            'text': transcription_text,
            'language': transcription_result.get('language', 'es'),
            'confidence': transcription_result.get('confidence', 0.0)
        },
        'voice_analysis': voice_analysis,
        'emotion_analysis': emotion_analysis,
        'content_analysis': content_analysis,
        'question_response_match': question_response_match,
        'answer_evaluation': answer_evaluation,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'transcription_confidence': transcription_result.get('confidence', 0.0),
            'voice_analysis_score': voice_analysis.get('score', 0),
            'emotion_stability': calculate_emotion_stability(emotion_analysis)
        }
    }

def calculate_emotion_stability(emotion_data):
    """Calcula un score de estabilidad emocional"""
    if not emotion_data or 'emotions' not in emotion_data:
        return 0.0
    
    emotions = emotion_data['emotions']
    neutral_score = emotions.get('neutral', 0)
    negative_emotions = sum(emotions.get(e, 0) for e in ['angry', 'sad', 'fear'])
    
    stability = (neutral_score * 0.7) + ((1 - negative_emotions) * 0.3)
    return round(stability, 2)

# Configuración inicial de la aplicación
def initialize_session():
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'skills' not in st.session_state:
        st.session_state.skills = []
    if 'emotion_queue' not in st.session_state:
        st.session_state.emotion_queue = queue.Queue()

# Inicialización de componentes
resume_parser = ResumeParser()
voice_analyzer = VoiceAnalyzer()
speech_to_text = SpeechToText()
content_matcher = EnhancedContentMatcher()
interview_bot = InterviewBot()
facial_analyzer = FacialEmotionAnalyzer()

# Interfaz principal
def main():
    st.title("Sistema de Análisis de Entrevistas")
    initialize_session()
    
    # Sidebar
    st.sidebar.header("Configuración de la Entrevista")
    position_desc = st.sidebar.text_area("Descripción del Puesto (Requerimientos, Habilidades)")
    resume_file = st.sidebar.file_uploader("Subir CV (PDF)", type=['pdf'])
    
    if position_desc:
        content_matcher.set_position_requirements(position_desc)
    
    # Procesar CV
    if resume_file:
        with st.spinner("Analizando CV..."):
            resume_path = save_uploaded_file(resume_file)
            resume_analysis = resume_parser.analyze_resume(resume_path)
            
            if resume_analysis:
                st.session_state.skills = resume_analysis['skills']
                st.success("Análisis de CV completado!")
                
                # Mostrar matching con el puesto
                if position_desc:
                    with st.expander("🔍 Coincidencia con el Puesto"):
                        match_result = content_matcher.analyze_position_match(st.session_state.skills)
                        st.write(f"**Porcentaje de Coincidencia:** {match_result['match_percentage']:.1f}%")
                        st.progress(match_result['match_percentage'] / 100)
                        
                        st.write("**Habilidades que Coinciden:**")
                        for skill in match_result['matching_skills']:
                            st.success(f"✓ {skill}")
                        
                        if match_result['missing_skills']:
                            st.write("**Habilidades Faltantes:**")
                            for skill in match_result['missing_skills']:
                                st.error(f"✗ {skill}")
                
                os.unlink(resume_path)
                
                # Generar preguntas
                if not st.session_state.questions:
                    st.session_state.questions = interview_bot.generate_questions(
                        st.session_state.skills, 
                        num_questions=5
                    )
                    st.session_state.current_question_index = 0
                    st.session_state.current_question = st.session_state.questions[0]
    
    # Sección de entrevista
    if st.session_state.skills:
        st.header("Entrevista")
        
        if st.session_state.current_question:
            st.write(f"**Pregunta {st.session_state.current_question_index + 1} de {len(st.session_state.questions)}**")
            st.write(st.session_state.current_question['question'])
            
            # Navegación entre preguntas
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Anterior") and st.session_state.current_question_index > 0:
                    st.session_state.current_question_index -= 1
                    st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                    st.rerun()
            with col3:
                if st.button("Siguiente ➡️") and st.session_state.current_question_index < len(st.session_state.questions) - 1:
                    st.session_state.current_question_index += 1
                    st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                    st.rerun()
            
            # Grabación y análisis
            st.subheader("Grabación de Respuesta")
            webrtc_ctx = webrtc_streamer(
                key="emotion-analysis",
                video_processor_factory=LiveEmotionVideoProcessor,
                media_stream_constraints={"video": True, "audio": True},
                async_processing=True,
            )
            
            recording_status = st.empty()
            emotion_placeholder = st.empty()
            
            if webrtc_ctx.state.playing:
                if "start_time" not in st.session_state:
                    st.session_state.start_time = time.time()
                    st.session_state.is_recording = True
                    recording_status.info("🔴 Grabando... Presiona STOP para finalizar")
                
                # Mostrar emoción detectada
                if not st.session_state.emotion_queue.empty():
                    emotion = st.session_state.emotion_queue.get()
                    dominant = emotion.get("dominant_emotion", "desconocida")
                    emotion_placeholder.write(f"**Emoción Detectada:** {dominant.capitalize()}")
            else:
                if st.session_state.get("is_recording"):
                    st.session_state.is_recording = False
                    recording_status.success("✅ Grabación completada. Analizando...")
                    
                    # Simular grabación de audio
                    audio_path = save_audio(record_audio(3))
                    
                    # Realizar análisis
                    analysis = analyze_response(
                        audio_path,
                        st.session_state.current_question,
                        st.session_state.skills,
                        facial_analyzer
                    )
                    
                    # Mostrar resultados
                    with st.expander("📊 Resultados del Análisis"):
                        st.write("**Transcripción:**", analysis['transcription'])
                        
                        st.write("**Coherencia Pregunta-Respuesta:**")
                        similarity = analysis['question_response_match']['question_response_similarity']
                        st.write(f"{similarity:.2f} - {'✅ Relevante' if analysis['question_response_match']['is_relevant'] else '⚠️ Poco relevante'}")
                        
                        st.write("**Análisis de Voz:**")
                        st.write(analysis['voice_analysis'])
                        
                        st.write("**Emoción Detectada:**")
                        st.write(analysis['emotion_analysis'])
                        
                        st.write("**Evaluación:**")
                        st.write(analysis['answer_evaluation'])
                    
                    st.session_state.analysis_results.append(analysis)
                    
                    if audio_path and os.path.exists(audio_path):
                        os.unlink(audio_path)
        
        # Mostrar historial
        if st.session_state.analysis_results:
            st.header("Historial de Respuestas")
            for i, result in enumerate(st.session_state.analysis_results):
                with st.expander(f"Respuesta {i+1} - {result['timestamp']}"):
                    st.write("**Pregunta:**", st.session_state.questions[i]['question'] if i < len(st.session_state.questions) else "N/A")
                    st.write("**Transcripción:**", result['transcription'])
                    st.write("**Evaluación:**", result['answer_evaluation'])

if __name__ == "__main__":
    main()
