import streamlit as st
import cv2
import os
import tempfile
import time
from ultralytics import YOLO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64

class StreamlitHumanDetector:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.current_stats = {}
        
    def load_model(self, model_path):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def process_video_streamlit(self, video_file, confidence_threshold=0.5):
        """Process uploaded video with progress tracking"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_video_path = tmp_file.name
        
        # Initialize video capture
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return None, None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize tracking variables
        frame_count = 0
        detection_history = []
        total_detections = 0
        
        # Create progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_placeholder = st.empty()
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = self.model(frame, conf=confidence_threshold)[0]
            
            # Count humans
            human_count = 0
            human_boxes = []
            
            if results.boxes is not None:
                for box in results.boxes:
                    if int(box.cls[0]) == 0:  # Human class
                        human_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        human_boxes.append((x1, y1, x2, y2, conf))
            
            # Update statistics
            detection_history.append(human_count)
            total_detections += human_count
            
            # Annotate frame
            annotated_frame = self.annotate_frame_ui(frame, human_boxes, human_count, frame_count, total_detections)
            out.write(annotated_frame)
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Update status every 10 frames
            if frame_count % 10 == 0:
                status_text.write(f"Processing frame {frame_count}/{total_frames} - Current humans: {human_count}")
                
                # Real-time stats update
                current_stats = {
                    'Frames Processed': frame_count,
                    'Total Detections': total_detections,
                    'Average per Frame': total_detections / frame_count if frame_count > 0 else 0,
                    'Current Frame Humans': human_count
                }
                
                with stats_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Frames Processed", frame_count)
                    col2.metric("Total Detections", total_detections)
                    col3.metric("Avg per Frame", f"{total_detections / frame_count:.1f}")
                    col4.metric("Current Frame", human_count)
        
        # Cleanup
        cap.release()
        out.release()
        os.unlink(temp_video_path)
        
        # Final statistics
        final_stats = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'average_humans_per_frame': total_detections / frame_count if frame_count > 0 else 0,
            'max_humans_in_frame': max(detection_history) if detection_history else 0,
            'min_humans_in_frame': min(detection_history) if detection_history else 0,
            'frames_with_humans': sum(1 for count in detection_history if count > 0),
            'detection_percentage': (sum(1 for count in detection_history if count > 0) / frame_count) * 100 if frame_count > 0 else 0,
            'detection_history': detection_history
        }
        
        return output_path, final_stats
    
    def annotate_frame_ui(self, frame, human_boxes, human_count, frame_count, total_detections):
        """Annotate frame with detection information"""
        annotated = frame.copy()
        
        # Draw bounding boxes
        for i, (x1, y1, x2, y2, conf) in enumerate(human_boxes):
            # Draw rectangle
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add confidence label
            label = f"Human {i+1}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (int(x1), int(y1) - label_size[1] - 5), 
                         (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(annotated, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add statistics overlay
        stats_text = [
            f"Frame: {frame_count}",
            f"Humans: {human_count}",
            f"Total: {total_detections}"
        ]
        
        # Background for stats
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Add text
        for i, text in enumerate(stats_text):
            cv2.putText(annotated, text, (15, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated

def create_detection_charts(stats):
    """Create visualization charts for detection statistics"""
    
    # Detection history chart
    if stats['detection_history']:
        fig_history = px.line(
            x=list(range(1, len(stats['detection_history']) + 1)),
            y=stats['detection_history'],
            title="Human Detection Count per Frame",
            labels={'x': 'Frame Number', 'y': 'Number of Humans Detected'}
        )
        fig_history.update_layout(showlegend=False)
        
        # Distribution chart
        detection_counts = stats['detection_history']
        unique_counts = list(set(detection_counts))
        count_frequency = [detection_counts.count(count) for count in unique_counts]
        
        fig_dist = px.bar(
            x=unique_counts,
            y=count_frequency,
            title="Distribution of Human Counts",
            labels={'x': 'Number of Humans', 'y': 'Number of Frames'}
        )
        
        return fig_history, fig_dist
    
    return None, None

def download_video(video_path):
    """Create download link for processed video"""
    with open(video_path, "rb") as file:
        video_bytes = file.read()
    
    b64_video = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64_video}" download="processed_video.mp4">Download Processed Video</a>'
    return href

def main():
    st.set_page_config(
        page_title="Human Detection System",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Human Detection System")
    st.markdown("Upload a thermal video to detect and count humans using AI")
    
    # Initialize detector
    detector = StreamlitHumanDetector()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "Custom Model"]
    )
    
    if model_option == "Custom Model":
        uploaded_model = st.sidebar.file_uploader("Upload Custom Model", type=['pt'])
        if uploaded_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
                tmp_model.write(uploaded_model.read())
                model_path = tmp_model.name
        else:
            model_path = "yolov8n.pt"
    else:
        model_path = model_option
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    # Load model
    if detector.load_model(model_path):
        st.sidebar.success(f"✅ Model loaded: {model_path}")
    else:
        st.sidebar.error("❌ Failed to load model")
        return
    
    # Main interface
    st.header("📤 Video Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a thermal video for human detection analysis"
    )
    
    if uploaded_file is not None:
        # Display video info
        st.subheader("📹 Video Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size / (1024*1024):.2f} MB")
        
        with col2:
            st.write(f"**Model:** {model_path}")
            st.write(f"**Confidence:** {confidence_threshold}")
        
        # Process button
        if st.button("🚀 Start Detection", type="primary"):
            st.header("🔄 Processing Video")
            
            start_time = time.time()
            
            # Process video
            output_path, stats = detector.process_video_streamlit(
                uploaded_file, 
                confidence_threshold
            )
            
            if output_path and stats:
                processing_time = time.time() - start_time
                
                st.success(f"✅ Processing completed in {processing_time:.2f} seconds")
                
                # Display results
                st.header("📊 Detection Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Frames",
                        stats['total_frames']
                    )
                
                with col2:
                    st.metric(
                        "Total Detections",
                        stats['total_detections']
                    )
                
                with col3:
                    st.metric(
                        "Average per Frame",
                        f"{stats['average_humans_per_frame']:.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Detection Success Rate",
                        f"{stats['detection_percentage']:.1f}%"
                    )
                
                # Additional statistics
                st.subheader("📈 Detailed Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Detection Summary:**")
                    st.write(f"• Maximum humans in single frame: {stats['max_humans_in_frame']}")
                    st.write(f"• Minimum humans in single frame: {stats['min_humans_in_frame']}")
                    st.write(f"• Frames with human detection: {stats['frames_with_humans']}")
                    st.write(f"• Frames without human detection: {stats['total_frames'] - stats['frames_with_humans']}")
                
                with col2:
                    st.write("**Processing Performance:**")
                    st.write(f"• Processing time: {processing_time:.2f} seconds")
                    st.write(f"• Processing speed: {stats['total_frames']/processing_time:.1f} fps")
                    st.write(f"• Model used: {model_path}")
                    st.write(f"• Confidence threshold: {confidence_threshold}")
                
                # Charts
                fig_history, fig_dist = create_detection_charts(stats)
                
                if fig_history and fig_dist:
                    st.subheader("📊 Detection Analysis Charts")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(fig_history, use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                # Download processed video
                st.subheader("💾 Download Results")
                
                try:
                    with open(output_path, "rb") as video_file:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_file.read(),
                            file_name="human_detection_output.mp4",
                            mime="video/mp4"
                        )
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")
                
                # Clean up temporary file
                try:
                    os.unlink(output_path)
                except:
                    pass
            
            else:
                st.error("❌ Error processing video")

if __name__ == "__main__":
    main()