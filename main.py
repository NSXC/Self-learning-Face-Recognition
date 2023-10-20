import cv2
import os
import random
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
def load_info_from_json(person_id):
    info_file_path = os.path.join(f"person_{person_id}", "info.json")
    if os.path.exists(info_file_path):
        with open(info_file_path, "r") as info_file:
            return json.load(info_file)
    return None


def TRAIN(known_people, vgg16_model, min_similarity):
    for person_id, data in known_people.items():
        data_list = []
        labels = []

        for img_path in os.listdir(f"person_{person_id}"):
            img = cv2.imread(os.path.join(f"person_{person_id}", img_path))
            if img is not None:
                if not img.size == 0:  
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    embedding = vgg16_model.predict(np.expand_dims(img, axis=0))
                    data_list.append(embedding)
                    labels.append(int(person_id))
                    print(f"Learned {img_path}")
                else:
                    print(f"Skipped empty image: {img_path}")
            else:
                print(f"Skipped unreadable image: {img_path}")

        if data_list and labels:
            mean_embedding = np.mean(data_list, axis=0)
            known_people[person_id]['embedding'] = mean_embedding
    accuracy = 0
    total = 0
    for person_id, data in known_people.items():
        for img_path in os.listdir(f"person_{person_id}"):
            img = cv2.imread(os.path.join(f"person_{person_id}", img_path))
            if img is not None:
                if not img.size == 0: 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))

                    embedding = vgg16_model.predict(np.expand_dims(img, axis=0))
                    
                    similarity = cosine_similarity(embedding, known_people[person_id]['embedding'])
                    if similarity[0][0] > min_similarity:
                        accuracy += 1
                    total += 1
                else:
                    print(f"Skipped empty image: {img_path}")
            else:
                print(f"Skipped unreadable image: {img_path}")

    if total > 0:
        print(f"Accuracy gained during retraining: {accuracy / total * 100:.2f}%")
        time.sleep(5)

    info_data = load_info_from_json(person_id)
    if info_data:
        known_people[person_id]['info'] = info_data

    return known_people


detector = MTCNN(min_face_size=80)

cap = cv2.VideoCapture(0)

known_people = {}
min_similarity = 0.8
person_id_counter = 1
vgg16_model = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg'
)

retraining_interval = 200
last_retraining_time = time.time()

while True:
    ret, frame = cap.read()

    faces = detector.detect_faces(frame)

    for result in faces:
        x, y, w, h = result['box']
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = tf.keras.applications.vgg16.preprocess_input(face.reshape(1, 224, 224, 3))

        recognized = False
        best_guess_name = None
        person_id = None

        for existing_person_id, data in known_people.items():
            if data['embedding'] is not None:
                similarity = cosine_similarity(
                    vgg16_model.predict(face), data['embedding']
                )
                if similarity[0][0] > min_similarity:
                    info = data['info']
                    print(f"Recognized: {info['name']}")
                    recognized = True
                    best_similarity = similarity[0][0]
                    best_guess_name = info['name']
                    person_id = existing_person_id

                    if best_similarity >= min_similarity:
                        person_folder = f"person_{person_id}"
                        os.makedirs(person_folder, exist_ok=True)
                        image_count = data['image_count']
                        image_count += 1
                        cv2.imwrite(f"{person_folder}/image_{image_count}.jpg", frame[y:y + h, x:x + w])
                        data['image_count'] = image_count
                        print(f"Took a picture of {best_guess_name}!")

        if not recognized:
            for existing_person_id, data in known_people.items():
                for detected_face_id, detected_face_data in data['detected_faces'].items():
                    similarity = cosine_similarity(vgg16_model.predict(face), detected_face_data)
                    if similarity[0][0] > min_similarity:
                        recognized = True
                        person_id = existing_person_id
                        break

            if not recognized:
                person_id = str(person_id_counter)
                person_id_counter += 1
            aa = input("IN DB [t/f]")
            if aa == "t":
                print("in db")
            else:
                name = input("Enter the name for this face: ")
                age = input("Enter the age for this face: ")
                gender = input("Enter the gender for this face: ")
                ethnicgroup = input("Enter Ethnic Group[White,Hispanic,Black,Asian,Asian Islander, ECT]: ")
                if person_id not in known_people:
                    known_people[person_id] = {
                        'embedding': vgg16_model.predict(face),
                        'info': {'id': person_id, 'name': name, 'age': age, 'gender': gender, 'ethnicgroup': ethnicgroup},
                        'image_count': 1,
                        'detected_faces': {}
                    }
                else:
                    known_people[person_id]['detected_faces'][str(random.randint(1, 1000))] = vgg16_model.predict(face)
                person_folder = f"person_{person_id}"
                os.makedirs(person_folder, exist_ok=True)
                info_file_path = os.path.join(person_folder, "info.json")
                with open(info_file_path, "w") as info_file:
                    info_data = known_people[person_id]['info']
                    json.dump(info_data, info_file, indent=4)

        current_time = time.time()
        if current_time - last_retraining_time >= retraining_interval:
            known_people = TRAIN(known_people, vgg16_model, min_similarity)
            last_retraining_time = current_time

    if 1 == 2:  
        break

cap.release()
cv2.destroyAllWindows()
