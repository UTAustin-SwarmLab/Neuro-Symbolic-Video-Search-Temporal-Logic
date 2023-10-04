META_TO_IMAGENET = {
    "person": ["groom", "scuba_diver", "ballplayer"], 
    "bicycle": ["mountain_bike"], 
    "car": [
        "sports_car", "limousine", "racer", "convertible", "jeep", 
        "cab", "Model_T", "amphibian", "trailer_truck", "fire_engine", 
        "garbage_truck", "tow_truck"
    ],
    "airplane": ["aircraft_carrier", "airliner", "warplane"],
    "bus": ["school_bus", "trolleybus"],
    "train": ["steam_locomotive"],
    "truck": [
        "trailer_truck", "fire_engine", "garbage_truck", 
        "moving_van", "tow_truck", "tank"
    ],
    "boat": [
        "canoe", "catamaran", "yawl", "gondola", "paddlewheel", 
        "schooner", "pirate", "airship", "submarine"
    ],
    "traffic light": ["traffic_light"],
    "fire hydrant": [],
    "stop sign": ["street_sign"],
    "parking meter": ["parking_meter"],
    "bench": ["park_bench"],
    "bird": [
        "cock", "hen", "ostrich", "brambling", "goldfinch", 
        "robin", "bulbul", "jay", "magpie", "chickadee", 
        "water_ouzel", "kite", "bald_eagle", "vulture", 
        "great_grey_owl", "European_fire_salamander", "common_newt", "eft"
    ],
    "cat": [
        "Egyptian_cat", "tiger_cat", "Persian_cat", "Siamese_cat", 
        "tabby", "lynx"
    ],
    "dog": [
        "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese", 
        "Shih-Tzu", "Blenheim_spaniel", "papillon", "toy_terrier", 
        "Rhodesian_ridgeback", "Afghan_hound", "basset", "beagle", 
        "bloodhound", "bluetick"
    ],
    "horse": ["sorrel", "zebra", "Appenzeller"],
    "sheep": [
        "ram", "bighorn", "ibex", "impala", "gazelle", 
        "Arabian_camel", "llama"
    ],
    "cow": ["ox", "water_buffalo", "bison"],
    "elephant": ["Indian_elephant", "African_elephant"],
    "bear": [
        "brown_bear", "American_black_bear", "ice_bear", "sloth_bear"
    ],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag"],
    "tie": ["tie", "Windsor_tie"],
    "suitcase": ["suitcase"],
    "frisbee": [],
    "skis": [],
    "snowboard": [],
    "sports ball": ["soccer_ball", "rugby_ball", "tennis_ball", "basketball"],
    "kite": ["kite"],
    "baseball bat": [],
    "baseball glove": [],
    "skateboard": [],
    "surfboard": [],
    "tennis racket": [],
    "bottle": ["water_bottle", "beer_bottle", "wine_bottle", "pop_bottle"],
    "wine glass": [],
    "cup": ["coffee_mug", "cup"],
    "fork": [],
    "knife": [],
    "spoon": [],
    "bowl": ["soup_bowl"],
    "banana": ["banana"],
    "apple": ["Granny_Smith"],
    "sandwich": [],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": [],
    "hot dog": ["hotdog"],
    "pizza": ["pizza"],
    "donut": [],
    "cake": [],
    "chair": ["rocking_chair", "folding_chair"],
    "couch": [],
    "potted plant": [],
    "bed": ["four-poster"],
    "dining table": ["dining_table"],
    "toilet": ["toilet_seat"],
    "tv": ["television"],
    "laptop": ["laptop"],
    "mouse": ["mouse"],
    "remote": ["remote_control"],
    "keyboard": ["computer_keyboard", "typewriter_keyboard"],
    "cell phone": ["cellular_telephone"],
    "microwave": ["microwave"],
    "oven": [],
    "toaster": ["toaster"],
    "sink": [],
    "refrigerator": ["refrigerator"],
    "book": ["book_jacket", "comic_book"],
    "clock": ["wall_clock", "analog_clock", "digital_clock"],
    "vase": ["vase"],
    "scissors": [],
    "teddy bear": ["teddy"],
    "hair drier": [],
    "toothbrush": ["toothbrush"]
}

# Since these classes are from GPT we write a filter to filter out any metaclasses 
# that are not in the imagenet classes

def filter(
        meta_to_imagenet:dict,
        imagenet_path:str
    ) -> dict:
    '''
    Returns the filtered META_TO_IMAGENET dictionary
    '''
    
    classes = set()
    text = ""
    for j,line in enumerate(open(imagenet_path)):
        cs = line[9:].strip().split(", ")[0]
        cs = cs.replace(" ", "_")
        classes.add(cs)
    filtered_meta_to_imagenet = dict()
    for k in meta_to_imagenet.keys():
        filtered_meta_to_imagenet[k] = []
        for v in meta_to_imagenet[k]:
            if v in classes:
                filtered_meta_to_imagenet[k].append(v)
    
    return filtered_meta_to_imagenet