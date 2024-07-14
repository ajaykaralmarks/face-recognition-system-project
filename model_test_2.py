import numpy as np
import cv2
import os
import sys
from tensorflow.keras.models import load_model
from gooey import Gooey, GooeyParser

# Load the pretrained model
model = load_model('celebrity_model2.h5')

# Define the image size (must be consistent with the training size)
img_size = (216, 216)

# Load the class indices (this should match the order of classes used during training)
train_data_dir = 'data1'
class_indices = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_data_dir)))}
reverse_class_indices = {idx: cls for cls, idx in class_indices.items()}

# Print class indices for debugging
print("Class Indices:", class_indices)
print("Reverse Class Indices:", reverse_class_indices)

# Define the face detection model (Haar Cascade classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@Gooey(program_name="Face Recognition", required_cols=1, optional_cols=1, default_size=(600, 400))
def main():
    parser = GooeyParser(description="Face Recognition using Pretrained Model")
    
    parser.add_argument('InputType', choices=['Image', 'Video', 'Camera'], help='Choose the input type')
    parser.add_argument('InputPath', widget='FileChooser', help='Select the input file (image or video)', gooey_options={'wildcard': "Images and Videos|*.jpg;*.jpeg;*.png;*.mp4;*.avi"})
    args = parser.parse_args()
    
    input_type = args.InputType
    input_path = args.InputPath
    
    if input_type == 'Image':
        process_image(input_path)
    elif input_type == 'Video':
        process_video(input_path)
    else:
        process_camera()

def process_image(image_path):
    # Load the image
    frame = cv2.imread(image_path)
    process_frame(frame, image_path, save_image=True)

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('runs/processed_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_camera():
    # Set the camera source (usually 0 for built-in webcam)
    camera_source = 0

    # Open a connection to the camera
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        process_frame(frame)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, image_path=None, save_image=False):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess the face image for model input
        face_img_resized = cv2.resize(face_img, img_size)
        face_img_normalized = face_img_resized / 255.0  # Normalize pixel values
        
        # Expand dimensions to match model input shape
        face_img_input = np.expand_dims(face_img_normalized, axis=0)
        
        # Predict the class
        prediction = model.predict(face_img_input)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_probability = np.max(prediction, axis=1)[0]
        
        # Debugging output
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Predicted probability: {predicted_probability}")
        
        # Only display the prediction if confidence is above 50%
        if predicted_probability > 0.50:
            if predicted_class_idx in reverse_class_indices:
                predicted_student_id = reverse_class_indices[predicted_class_idx]
                
                # Draw rectangle around the face and display predicted student ID
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Student ID: {predicted_student_id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f'Confidence: {predicted_probability:.2f}', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print(f"Warning: Predicted class index {predicted_class_idx} not found in reverse_class_indices")
        else:
            # Optional: Draw a different rectangle or message if confidence is below 50%
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Uncertain', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    if save_image and image_path:
        save_path = os.path.join('runs', os.path.basename(image_path))
        cv2.imwrite(save_path, frame)
    
    return frame

if __name__ == '__main__':
    main()
