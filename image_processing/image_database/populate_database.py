import pymongo
import GridFS

# establish mongo client connection
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")

# define image database for the client
image_database = myclient["image_database"]
# initialize GridFS for storing images as object ID's
fs = gridfs.GridFS(image_database)


# checks for the existence of the database
# database_list = client.list_database_names()
#
# if "image_database" in database_list:
#     print("The database exists.")
# else:
#     print("The database does not exist")

def insert_into_database(collection_dict):

    # initialize a collection for the image database
    image_collection = image_database["image_collection"]
    
    # convert input image ndarray to string
    input_image_string = collection_dict["image"].tostring()
    # convert input image string to an object ID
    input_image_ID = fs.put(imageString, encoding='utf-8')
    collection_dict["image"] = input_image_ID
    
    # convert detected objects
    if collection_dict["detected_objects"]:
        objects = []
        for detected_object in collection_dict["detected_objects"]:
            # convert detected object ndarray to string
            detected_object_string = detected_object.tostring()
            # convert detected object string to an object ID
            detected_object_ID = fs.put(detected_object_string, encoding='utf-8')
            objects.append(detected_object_ID)
        collection_dict["detected_objects"] = objects
        
    # insert collection into the database
    image_collection.insert_one(collection_dict)