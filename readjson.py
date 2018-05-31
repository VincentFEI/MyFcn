import json

# params_dict = {
#     "IMAGE_WIDTH" : 1024,
#     "IMAGE_HEIGHT" : 2048,
#     "BATCH_SIZE" : 2,
#     "NUM_CLASSES" : 21,
#     "MAX_ITERATION" : 22,
#     "LEARNING_RATE" : 1e-4,
#     "KEEP_PROBABILITY" : 0.85,
#     "DEBUG" : True,
#     "MODE" : "train",
#     "LOGS_DIR" : "logs/"
# }


with open('params.json', 'r') as f:
    params_dict = json.load(f)

print(params_dict)
print(params_dict["KEEP_PROBABILITY"])