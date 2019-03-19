from flask import Flask
from flask_restful import Resource, Api
import json
import boto3
import botocore
import os
import array 
import datetime
from imageai.Detection import ObjectDetection
from subprocess import call
    
# S3
s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

# ImageAI
execution_path = os.getcwd()

# S3
BUCKET = 'hubquarters'
SOURCE_FOLDER = 'source/'
DESTINATION_FOLDER = 'processed/'

app = Flask(__name__)
api = Api(app)

def abortIfImageDoesNotExist(rawImageName):
    rawImage = "test"
    if not rawImage:
        abort(404, message="Image does not exist")

class PeopleCounter(Resource):
    def get(self, rawImageName):
        # Abort if image does not exist
        abortIfImageDoesNotExist(rawImageName)

        # Get the rawImage from Amazon s3
        try:
            s3_resource.Bucket(BUCKET).download_file(SOURCE_FOLDER + rawImageName, os.path.join(execution_path, "temp/rawImage.jpg"))
        except botocore.exceptions.ClientError as e:
            return e.response['Error']

        #  countNumPeople in rawImage
        if os.path.isfile(os.path.join(execution_path, "temp/rawImage.jpg")):
            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
            detector.loadModel()
            personOnlyModel = detector.CustomObjects(person=True)
            detections = detector.detectCustomObjectsFromImage(
                custom_objects=personOnlyModel, 
                input_image=os.path.join(execution_path, "temp/rawImage.jpg"), 
                output_image_path=os.path.join(execution_path, "temp/processedImageName.jpg"))

        # Upload processed temp file to s3
        s3_client.upload_file(os.path.join(execution_path, "temp/processedImageName.jpg"), BUCKET, DESTINATION_FOLDER + rawImageName, ExtraArgs={'ACL':'public-read'})
        call('rm -rf temp/*', shell=True)

        # Tenants Probability
        tenants = []
        for eachObject in detections:
            tenants.append({eachObject["name"]:eachObject["percentage_probability"]})

        # Current time
        currentDT = datetime.datetime.now()

        # Do the processing
        return {
            'statusCode': 200,
            'processedFileName': rawImageName,
            'tenantsDetected': tenants,
            'totalDetected': len(detections),
            'timeDetetected' : currentDT.strftime("%d %b, %Y - %I:%M:%S %p")
        }

api.add_resource(PeopleCounter, '/api/peoplecounter/<string:rawImageName>')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)