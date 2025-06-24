# Thermal Human Detection using YOLOv8

This project detects humans in thermal CCTV videos using a YOLOv8 model, optimized for real-time usage and intended for integration into Hand-Held Thermal Imaging (HHTI) systems.

## 📁 Project Structure
thermal-human-detection/
├── input_video/           # Thermal video to test
├── output_video/          # Annotated output video
├── output_images/         #  output frames
├── thermal/               # YOLOv8 dataset (train/val + data.yaml)
├── yolov8n.pt             # Base YOLO model
├── train.py               # Model training script
├── source.py              # Script to run inference on a thermal video
├── interface.py           # Streamlit-based UI to test model on video
├── report.docx            # Final summary report
├── README.md              # Instructions
└── requirements.txt       # Dependency list

pip install -r requirements.txt