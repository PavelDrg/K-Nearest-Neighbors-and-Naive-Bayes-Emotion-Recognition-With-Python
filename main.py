import dlib
from skimage import io
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
import numpy as np


def extract_features_labels(data_folder, face_detector, shape_predictor):
    X, y = [], []

    # Define a mapping of emotion folder names to labels
    emotion_mapping = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "surprise": "surprise"
    }

    for emotion_folder in os.listdir(data_folder):
        emotion_path = os.path.join(data_folder, emotion_folder)
        if os.path.isdir(emotion_path):
            print(f"Processing folder: {emotion_folder}")
            if not os.listdir(emotion_path):
                print(f"Empty folder: {emotion_folder}")
            for image_file in os.listdir(emotion_path):
                image_path = os.path.join(emotion_path, image_file)
                print(f"Processing image: {image_path}")

                # Load image using dlib
                img = io.imread(image_path)

                # Detect faces in the image
                faces = face_detector(img)

                for face in faces:
                    # Get facial landmarks
                    shape = shape_predictor(img, face)

                    # Compute HOG features for the detected face
                    hog_features = np.ravel(dlib.get_face_hog(img, shape))

                    # Use the emotion mapping to assign the label
                    label = emotion_mapping.get(emotion_folder.lower(), "unknown")

                    # Append features and label
                    X.append(hog_features)
                    y.append(label)
                    print(f"Detected label: {label}")

    # Debug information
    unique_labels = set(y)
    print(f"Number of unique labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")

    return np.array(X), np.array(y)



def main():
    data_folder = "dataset"

    # Load pre-trained face detector and shape predictor models
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor_path = r'D:\Proiecte TIA\ZUZUZU\shape_predictor_68_face_landmarks.dat'
    shape_predictor = dlib.shape_predictor(shape_predictor_path)

    # Extract features and labels
    X, y = extract_features_labels(data_folder, face_detector, shape_predictor)

    # Debug information
    unique_labels = set(y)
    print(f"Number of unique labels: {len(unique_labels)}")
    print(f"Unique labels: {unique_labels}")

    # Check if there are enough samples for splitting
    if len(unique_labels) <= 1:
        print("Not enough samples for splitting. Please provide more data.")
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a support vector machine (SVM) classifier
    classifier = svm.SVC(kernel='linear', C=1, probability=True)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = classifier.predict(X_test)

    # Evaluate the performance
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()