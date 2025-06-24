import cv2
from ultralytics import YOLO
import os
import numpy as np
from collections import defaultdict
import time

class HumanDetectionCounter:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.frame_count = 0
        self.total_detections = 0
        self.detection_history = []
        self.confidence_threshold = 0.5
        
    def process_video(self, video_path, output_path="output_video/annotated_output.mp4"):
        """Process video with human detection and counting"""
        os.makedirs("output_video", exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        print(f"🎥 Processing video: {video_path}")
        print(f"📊 Video info: {w}x{h} @ {fps}fps, {total_frames} frames")
        print("-" * 50)
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold)[0]
            
            # Count humans (class 0 in COCO dataset)
            human_count = 0
            human_boxes = []
            
            if results.boxes is not None:
                for box in results.boxes:
                    if int(box.cls[0]) == 0:  # Human class
                        human_count += 1
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        human_boxes.append((x1, y1, x2, y2, conf))
            
            # Update statistics
            self.detection_history.append(human_count)
            self.total_detections += human_count
            
            # Create annotated frame
            annotated_frame = self.annotate_frame(frame, human_boxes, human_count)
            
            # Write frame
            out.write(annotated_frame)
            
            # Progress update
            if self.frame_count % 30 == 0:  # Every 30 frames
                progress = (self.frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Frame: {self.frame_count}/{total_frames} | Current humans: {human_count}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate and display statistics
        self.display_statistics(video_path, output_path, time.time() - start_time)
        
        return self.get_statistics()
    
    def annotate_frame(self, frame, human_boxes, human_count):
        """Add annotations to frame with human detection info"""
        annotated = frame.copy()
        
        # Draw bounding boxes
        for i, (x1, y1, x2, y2, conf) in enumerate(human_boxes):
            # Draw rectangle
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add confidence label
            label = f"Human {i+1}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(annotated, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add frame statistics overlay
        stats_text = [
            f"Frame: {self.frame_count}",
            f"Humans Detected: {human_count}",
            f"Total Detections: {self.total_detections}",
            f"Avg per Frame: {self.total_detections/self.frame_count:.1f}"
        ]
        
        # Background for stats
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Add text
        for i, text in enumerate(stats_text):
            cv2.putText(annotated, text, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def get_statistics(self):
        """Return comprehensive statistics"""
        if not self.detection_history:
            return {}
        
        return {
            'total_frames': self.frame_count,
            'total_detections': self.total_detections,
            'average_humans_per_frame': self.total_detections / self.frame_count,
            'max_humans_in_frame': max(self.detection_history),
            'min_humans_in_frame': min(self.detection_history),
            'frames_with_humans': sum(1 for count in self.detection_history if count > 0),
            'frames_without_humans': sum(1 for count in self.detection_history if count == 0),
            'detection_percentage': (sum(1 for count in self.detection_history if count > 0) / self.frame_count) * 100
        }
    
    def display_statistics(self, video_path, output_path, processing_time):
        """Display comprehensive statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("🎯 HUMAN DETECTION ANALYSIS COMPLETE")
        print("="*60)
        print(f"📹 Input Video: {video_path}")
        print(f"💾 Output Video: {output_path}")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"🚀 Processing Speed: {self.frame_count/processing_time:.1f} fps")
        print("\n📊 DETECTION STATISTICS:")
        print("-" * 30)
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Total Human Detections: {stats['total_detections']}")
        print(f"Average Humans per Frame: {stats['average_humans_per_frame']:.2f}")
        print(f"Maximum Humans in Single Frame: {stats['max_humans_in_frame']}")
        print(f"Minimum Humans in Single Frame: {stats['min_humans_in_frame']}")
        print(f"Frames with Human Detection: {stats['frames_with_humans']}")
        print(f"Frames without Human Detection: {stats['frames_without_humans']}")
        print(f"Detection Success Rate: {stats['detection_percentage']:.1f}%")
        print("="*60)
        print("✅ Analysis saved to annotated video!")
    
    def save_statistics_report(self, output_dir="output_video"):
        """Save detailed statistics to text file"""
        stats = self.get_statistics()
        report_path = os.path.join(output_dir, "detection_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("HUMAN DETECTION ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Used: {self.model.model_name}\n")
            f.write(f"Confidence Threshold: {self.confidence_threshold}\n\n")
            
            f.write("DETECTION STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write(f"\nFrame-by-frame Detection Count:\n")
            f.write("-" * 30 + "\n")
            for i, count in enumerate(self.detection_history, 1):
                f.write(f"Frame {i}: {count} humans\n")
        
        print(f"📄 Detailed report saved to: {report_path}")

# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = HumanDetectionCounter("yolov8n.pt")  # Replace with your trained model
    
    # Process video
    video_path = "input video/1721303-hd_1920_1080_25fps.mp4"  # Replace with your video
    
    try:
        stats = detector.process_video(video_path)
        detector.save_statistics_report()
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        print("Please check your video path and model file.")