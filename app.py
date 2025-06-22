import streamlit as st
import os
# Importar y aplicar el parche para LocallyConnected2D antes de cargar DeepFace
import keras_bridge
keras_bridge.install_patch()
from resume_parser import ResumeParser

# Try to import optional modules
try:
    from facial_emotion import FacialEmotionAnalyzer
    FACIAL_EMOTION_AVAILABLE = True
except Exception as e:
    st.warning(f"Analsis facial no disponible: {str(e)}")
    FACIAL_EMOTION_AVAILABLE = False
    FacialEmotionAnalyzer = None

from voice_analysis import VoiceAnalyzer
# Importar SpeechToText directamente del módulo
from speech_to_text import SpeechToText
from content_matcher import ContentMatcher
from interview_bot import InterviewBot
import tempfile
import time
import cv2
import numpy as np
import threading
import queue
import sounddevice as sd
import wave
import json
from datetime import datetime

# Initialize components
resume_parser = ResumeParser()
# Forzar la inicialización del analizador facial
try:
    facial_analyzer = FacialEmotionAnalyzer()
    FACIAL_EMOTION_AVAILABLE = True
except Exception as e:
    print(f"Error al inicializar el analizador facial: {str(e)}")
    facial_analyzer = None
    FACIAL_EMOTION_AVAILABLE = False

voice_analyzer = VoiceAnalyzer()
speech_to_text = SpeechToText()
content_matcher = ContentMatcher()
interview_bot = InterviewBot()

# Global variables for state management
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

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def process_frame(frame, emotion_queue):
    """Process a single frame for emotion analysis."""
    try:
        if facial_analyzer:
            print(f"🔍 DEBUG: Processing frame with facial_analyzer available")
            analysis = facial_analyzer.analyze_frame(frame)
            print(f"🔍 DEBUG: Frame analysis result: {analysis}")
            # Ensure the analysis has the required keys
            if 'dominant_emotion' not in analysis:
                analysis['dominant_emotion'] = 'neutral'
                print(f"⚠️ WARNING: No dominant_emotion found, defaulting to neutral")
            if 'emotions' not in analysis:
                analysis['emotions'] = {}
                print(f"⚠️ WARNING: No emotions found, defaulting to empty dict")
            emotion_queue.put(analysis)
        else:
            print(f"❌ ERROR: facial_analyzer not available")
            emotion_queue.put({'dominant_emotion': 'unavailable', 'emotions': {}})
    except Exception as e:
        print(f"❌ ERROR processing frame: {str(e)}")
        import traceback
        print(f"❌ ERROR traceback: {traceback.format_exc()}")
        # Put a default emotion result even when there's an error
        emotion_queue.put({'dominant_emotion': 'error', 'emotions': {}})

def record_audio(duration, sample_rate=44100):
    """Record audio for a specified duration."""
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return recording

def save_audio(recording, sample_rate=44100):
    """Save recorded audio to a temporary WAV file."""
    try:
        # Create a temporary file with explicit close to avoid Windows file locking
        import uuid
        temp_filename = f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Ensure recording is not empty and has valid data
        if len(recording) == 0:
            print("Warning: Empty recording")
            return None
            
        # Normalize audio data
        audio_data = np.array(recording).flatten()
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
        # Save with explicit file handling
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        # Verify file was created
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 44:  # More than just header
            return temp_path
        else:
            print(f"Error: Audio file not created properly: {temp_path}")
            return None
            
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return None

def analyze_response(audio_path, video_frames, question, skills):
    """Analyze the user's response comprehensively."""
    # Initialize default results
    transcription = {'text': '', 'segments': [], 'language': 'es'}
    voice_analysis = {}
    
    # Check if audio file exists and is valid
    if audio_path and os.path.exists(audio_path):
        try:
            # Transcribe speech
            transcription = speech_to_text.transcribe(audio_path)
            
            # Analyze voice characteristics
            voice_features = voice_analyzer.extract_features(audio_path)
            voice_analysis = voice_analyzer.analyze_voice_characteristics(voice_features)
        except Exception as e:
            print(f"Error in audio analysis: {str(e)}")
            transcription = {'text': 'Error processing audio', 'segments': [], 'language': 'en'}
            voice_analysis = {'error': 'Audio analysis failed'}
    else:
        print(f"Audio file not available: {audio_path}")
        transcription = {'text': 'Audio recording failed', 'segments': [], 'language': 'en'}
        voice_analysis = {'error': 'No audio file'}
    
    # Analyze facial emotions
    if facial_analyzer and video_frames:
        try:
            print(f"🔍 DEBUG: Analyzing {len(video_frames)} video frames for final emotion summary")
            emotion_analysis = facial_analyzer.analyze_frames(video_frames)
            print(f"🔍 DEBUG: Final emotion analysis result: {emotion_analysis}")
            # Ensure the analysis has the required keys
            if not isinstance(emotion_analysis, dict):
                emotion_analysis = {'dominant_emotion': 'error', 'emotions': {}}
                print(f"⚠️ WARNING: emotion_analysis is not a dict, got: {type(emotion_analysis)}")
            if 'dominant_emotion' not in emotion_analysis:
                emotion_analysis['dominant_emotion'] = 'neutral'
                print(f"⚠️ WARNING: No dominant_emotion in final analysis")
            if 'emotions' not in emotion_analysis:
                emotion_analysis['emotions'] = {}
                print(f"⚠️ WARNING: No emotions in final analysis")
        except Exception as e:
            print(f"❌ ERROR in emotion analysis: {str(e)}")
            import traceback
            print(f"❌ ERROR traceback: {traceback.format_exc()}")
            emotion_analysis = {'dominant_emotion': 'error', 'emotions': {}}
    else:
        print(f"❌ ERROR: facial_analyzer={facial_analyzer is not None}, video_frames={len(video_frames) if video_frames else 0}")
        emotion_analysis = {'dominant_emotion': 'unavailable', 'emotions': {}}
    
    # Match content with resume
    content_analysis = content_matcher.analyze_content_match(skills, transcription['text'])
    
    # Evaluate answer
    answer_evaluation = interview_bot.evaluate_answer(question, transcription['text'], skills)
    
    return {
        'transcription': transcription['text'],
        'voice_analysis': voice_analysis,
        'emotion_analysis': emotion_analysis,
        'content_analysis': content_analysis,
        'answer_evaluation': answer_evaluation,
        'timestamp': datetime.now().isoformat()
    }

def main():
    st.title("Sistema de analisis de entrevistas")
    
    # Sidebar for resume upload
    st.sidebar.header("Subir CV")
    resume_file = st.sidebar.file_uploader("Subir(PDF)", type=['pdf'])
    
    # Mostrar estado del análisis facial
    if FACIAL_EMOTION_AVAILABLE:
        st.sidebar.success("✅ Análisis facial disponible")
    else:
        st.sidebar.error("❌ Análisis facial no disponible")
        st.sidebar.info("Intente reiniciar la aplicación o verificar la cámara")
    
    # Main interview interface
    st.header("Entrevista en vivo")
    
    if resume_file:
        # Process resume
        with st.spinner("Analizando CV..."):
            resume_path = save_uploaded_file(resume_file)
            resume_analysis = resume_parser.analyze_resume(resume_path)
            if resume_analysis:
                st.session_state.skills = resume_analysis['skills']
                st.success("Analisis de CV completado!")
                st.sidebar.write("Extracted Skills:", resume_analysis['skills'])
                os.unlink(resume_path)
                
                # Generar múltiples preguntas inmediatamente después de procesar el CV
                try:
                    if 'questions' not in st.session_state or not st.session_state.questions:
                        st.session_state.questions = interview_bot.generate_questions(
                            st.session_state.skills, 
                            num_questions=8  # Generar 8 preguntas en lugar de 1
                        )
                        st.session_state.current_question_index = 0
                        st.session_state.current_question = st.session_state.questions[0]
                except Exception as e:
                    st.error(f"Error al generar preguntas: {str(e)}")
            else:
                st.error("Error al analizar CV. Revisar formato.")
    
    # Display current question - SIEMPRE mostrar si existe
    if 'skills' in st.session_state and st.session_state.skills:
        # Si no hay preguntas pero hay skills, generar múltiples preguntas
        if 'questions' not in st.session_state or not st.session_state.questions:
            try:
                st.session_state.questions = interview_bot.generate_questions(
                    st.session_state.skills, 
                    num_questions=8  # Generar 8 preguntas
                )
                st.session_state.current_question_index = 0
                st.session_state.current_question = st.session_state.questions[0]
            except Exception as e:
                st.error(f"Error al generar preguntas: {str(e)}")
        
        # Mostrar la pregunta actual y el progreso
        if st.session_state.current_question:
            st.write(f"### Pregunta {st.session_state.current_question_index + 1} de {len(st.session_state.questions)}")
            st.write(st.session_state.current_question['question'])
            
            # Mostrar todas las preguntas generadas en un expander
            with st.expander("Ver todas las preguntas generadas"):
                for i, q in enumerate(st.session_state.questions):
                    status = "✅ Completada" if i < st.session_state.current_question_index else "🔄 Actual" if i == st.session_state.current_question_index else "⏳ Pendiente"
                    st.write(f"{i+1}. {q['question']} - {status}")
            
            # Botones de navegación
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Pregunta anterior") and st.session_state.current_question_index > 0:
                    st.session_state.current_question_index -= 1
                    st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                    st.rerun()
            
            with col3:
                if st.button("Pregunta siguiente ➡️") and st.session_state.current_question_index < len(st.session_state.questions) - 1:
                    st.session_state.current_question_index += 1
                    st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                    st.rerun()
            
            # Start/Stop recording button - SIEMPRE mostrar si hay pregunta
            if st.button("Comenzar grabacion", use_container_width=True):
                st.session_state.is_recording = True
                
                # Mostrar indicador de grabación activa
                recording_status = st.empty()
                recording_status.error("🔴 GRABANDO... Por favor responde a la pregunta")
                
                # Create placeholders for live feedback
                emotion_placeholder = st.empty()
                voice_placeholder = st.empty()
                
                # Initialize video capture
                try:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("❌ No se pudo acceder a la cámara. Verifique que esté conectada y no esté siendo usada por otra aplicación.")
                        st.session_state.is_recording = False
                        return
                except Exception as e:
                    st.error(f"❌ Error al inicializar la cámara: {str(e)}")
                    st.session_state.is_recording = False
                    return
                    
                video_frames = []
                emotion_queue = queue.Queue()
                
                # Record for 30 seconds
                start_time = time.time()
                frame_count = 0
                while time.time() - start_time < 30 and st.session_state.is_recording:
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                        video_frames.append(frame)
                        
                        # Procesar cada 5 frames para evitar sobrecarga
                        if frame_count % 5 == 0:
                            print(f"🔍 DEBUG: Processing frame {frame_count}")
                            # Process frame directly instead of in thread for debugging
                            process_frame(frame, emotion_queue)
                        
                        # Actualizar tiempo restante
                        elapsed_time = time.time() - start_time
                        remaining_time = 30 - elapsed_time
                        recording_status.error(f"🔴 GRABANDO... Tiempo restante: {remaining_time:.1f} segundos")
                        
                        # Display live emotion analysis
                        if not emotion_queue.empty():
                            emotion = emotion_queue.get()
                            # Safely access dominant_emotion with a fallback
                            dominant_emotion = emotion.get('dominant_emotion', 'unknown')
                            emotion_placeholder.write(f"Emocion Actual: {dominant_emotion}")
                            print(f"🔍 DEBUG: Emoción detectada en tiempo real: {dominant_emotion}")
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.1)
                
                print(f"🔍 DEBUG: Grabación completada. Total frames: {len(video_frames)}")
                recording_status.success("✅ Grabación completada - Analizando respuesta...")
                
                cap.release()
                
                # Record audio
                audio_data = record_audio(30)
                audio_path = save_audio(audio_data)
                
                # Analyze response
                with st.spinner("Analizando tu respuesta..."):
                    analysis = analyze_response(
                        audio_path,
                        video_frames,
                        st.session_state.current_question,
                        st.session_state.skills
                    )
                    
                    st.session_state.analysis_results.append(analysis)
                    
                    # Display analysis results
                    st.write("### Resultado del analisis")

                    st.write("#### Transcripcion")
                    print(f"🟡 Texto transcrito: '{analysis['transcription']}'")
                    if not analysis['transcription'].strip():
                        st.warning("⚠️ No se obtuvo ninguna transcripción del audio.")
                    else:
                        st.write(analysis['transcription'])
                    
                    st.write("#### Analisis de voz")
                    st.write(analysis['voice_analysis'])
                    
                    st.write("#### Analisis de emociones")
                    st.write(analysis['emotion_analysis'])
                    
                    st.write("#### Contenido similar")
                    st.write(analysis['content_analysis'])
                    
                    st.write("#### Evaluacion de la respuesta")
                    st.write(analysis['answer_evaluation'])
                
                # Clean up
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                st.session_state.is_recording = False
                
                # Avanzar a la siguiente pregunta
                if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                    st.session_state.current_question_index += 1
                    st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                    st.success(f"Avanzando a la pregunta {st.session_state.current_question_index + 1}")
                else:
                    st.success("🎉 ¡Has completado todas las preguntas de la entrevista!")
                    st.balloons()
                    # Reiniciar si se desea continuar
                    if st.button("Generar nuevas preguntas"):
                        st.session_state.questions = interview_bot.generate_questions(
                            st.session_state.skills,
                            num_questions=8
                        )
                        st.session_state.current_question_index = 0
                        st.session_state.current_question = st.session_state.questions[0]
    
    # Display interview history
    if st.session_state.analysis_results:
        st.header("Historial de entrevista")
        for i, result in enumerate(st.session_state.analysis_results):
            with st.expander(f"Response {i+1}"):
                st.write("Pregunta:", st.session_state.current_question['question'])
                st.write("Transcripcion:", result['transcription'])
                st.write("Evaluaacion:", result['answer_evaluation'])

if __name__ == "__main__":
    main() 