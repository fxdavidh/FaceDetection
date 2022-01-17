import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    sub_dir = os.listdir(root_path)
    
    return sub_dir

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''

    train_images = []
    class_indices = []

    for id,train_name in enumerate(train_names):
        path = '{}/{}'.format(root_path, train_name)
        images = os.listdir(path)

        for image in images:
            full_path = '{}/{}'.format(path, image)
            train_image = cv.imread(full_path)
            train_images.append(train_image)
            class_indices.append(id)
    
    return train_images, class_indices

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

    filtered_face_images = []
    filtered_rectangle_face_images = []
    face_indices = []

    face_detection = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    for id, image in enumerate(image_list):
        if not image_classes_list:
            gray_image = image['gray_image']
        else:
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=3)

        if len(faces) > 0:
            for face in faces:
                x,y,w,h = face
                filtered_face_image = gray_image[y:y+h, x:x+w]
                filtered_face_images.append(filtered_face_image)
                filtered_rectangle_face_images.append(face)
                if not image_classes_list:
                    continue
                else:
                    face_indices.append(image_classes_list[id])
    
    return filtered_face_images, filtered_rectangle_face_images, face_indices

def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    model = cv.face.LBPHFaceRecognizer_create()
    model.train(train_face_grays, np.array(image_classes_list))

    return model

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''

    gray_images = []

    images = os.listdir(test_root_path)

    for id, image in enumerate(images):
        full_path = '{}/{}'.format(test_root_path, image)
        image = cv.imread(full_path)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        temp = {'image':image,'gray_image':gray_image}
        gray_images.append(temp)

    return gray_images

def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    results = []
    for face in test_faces_gray:
        result = recognizer.predict(face)
        results.append(result)
    return results


def get_verification_status(prediction_result, train_names, unverified_names):
    '''
        To generate a list of verification status from prediction results

        Parameters
        ----------
        prediction_result : list
            List containing all prediction results from given test faces
        train_names : list
            List containing all loaded test images
        unverified_names : list
            List containing all unverified names
        
        Returns
        -------
        list
            List containing all verification status from prediction results
    '''
    status = []
    for i,pred in enumerate(prediction_result):
        if train_names[pred[0]] in unverified_names:
            temp = {'name': train_names[pred[0]], 'status':'unverified'}
            status.append(temp)
        else:
            temp = {'name': train_names[pred[0]], 'status':'verified'}
            status.append(temp)
    return status

def draw_prediction_results(verification_statuses, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results and verification status on the given test images

        Parameters
        ----------
        verification_statuses : list
            List containing all checked results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn
    '''
    drawn_images=[]

    for i,image in enumerate(test_image_list):
        x,y,w,h = test_faces_rects[i]
        if verification_statuses[i]['status'] == 'unverified':
            face_image = cv.rectangle(image['image'], (x,y), (x+w, y+h), (0,0,255))
            cv.putText(face_image, verification_statuses[i]['name'],(x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv.putText(face_image, verification_statuses[i]['status'], (x, h+y+20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        else:
            face_image = cv.rectangle(image['image'], (x,y), (x+w, y+h), (0,255,0))
            cv.putText(face_image, verification_statuses[i]['name'],(x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv.putText(face_image, verification_statuses[i]['status'], (x, h+y+20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        temp = {'image': face_image, 'status':verification_statuses[i]['status']}
        drawn_images.append(temp)

    return drawn_images

def combine_and_show_result(image_list):
    '''
        To show the final image that already combined into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    size = 1
    for i,data in enumerate(image_list):
        if data['status'] == 'unverified':
            plt.subplot(2, 3, size)
            plt.axis('off')
            data['image'] = cv.cvtColor(data['image'], cv.COLOR_BGR2RGB)
            plt.imshow(data['image'])
            size+=1

    for i,data in enumerate(image_list):
        if data['status'] == 'verified':
            plt.subplot(2, 3, size)
            plt.axis('off')
            data['image'] = cv.cvtColor(data['image'], cv.COLOR_BGR2RGB)
            plt.imshow(data['image'])
            size+=1

    plt.show()
'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "Dataset/Train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "Dataset/Test"
    unverified_names = ["Raditya Dika", "Anya Geraldine", "Raffi Ahmad"]

    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    prediction_result = predict(recognizer, test_faces_gray)
    verification_statuses = get_verification_status(prediction_result, train_names, unverified_names)
    predicted_test_image_list = draw_prediction_results(verification_statuses, test_image_list, test_faces_rects, train_names)
    
    combine_and_show_result(predicted_test_image_list) 