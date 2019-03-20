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
import gc
    
# S3
s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

# S3
BUCKET = 'hubquarters'
SOURCE_FOLDER = 'source/'
DESTINATION_FOLDER = 'processed/'

# ImageAI
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
personOnlyModel = detector.CustomObjects(person=True)

app = Flask(__name__)
api = Api(app)

class PeopleCounter(Resource):
    def get(self, rawImageName):
        # Get the rawImage from Amazon s3
        try:
            s3_resource.Bucket(BUCKET).download_file(SOURCE_FOLDER + rawImageName, os.path.join(execution_path, "temp/rawImage-" + rawImageName))
        except botocore.exceptions.ClientError as e:
            return e.response['Error']

        #  countNumPeople in rawImage
        if os.path.isfile(os.path.join(execution_path, "temp/rawImage-" + rawImageName)):
            detections = detector.detectCustomObjectsFromImage(
                custom_objects=personOnlyModel, 
                input_image=os.path.join(execution_path, "temp/rawImage-" + rawImageName), 
                output_image_path=os.path.join(execution_path, "temp/processedImageName-" + rawImageName))

        # Upload processed temp file to s3
        s3_client.upload_file(os.path.join(execution_path, "temp/processedImageName-" + rawImageName), BUCKET, DESTINATION_FOLDER + rawImageName, ExtraArgs={'ACL':'public-read'})

        # Tenants Probability
        tenants = []
        for eachObject in detections:
            tenants.append({eachObject["name"]:eachObject["percentage_probability"]})

        # Remove unused raw images and clear garbage collector
        call('rm -rf temp/*', shell=True)
        gc.collect()
        del(detections)

        # Current time
        currentDT = datetime.datetime.now()

        # Do the processing
        return {
            'statusCode': 200,
            'processedFileName': rawImageName,
            'tenantsDetected': tenants,
            'totalDetected': len(tenants),
            'timeDetetected' : currentDT.strftime("%d %b, %Y - %I:%M:%S %p")
        }

        os.abort()

api.add_resource(PeopleCounter, '/api/peoplecounter/<string:rawImageName>')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, threaded=False)