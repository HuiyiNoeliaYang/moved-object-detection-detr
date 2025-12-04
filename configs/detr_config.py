CLASS_ID_TO_NAME = {
    0: "Unknown",
    1: "person",
    2: "car",
    3: "other_vehicle",
    4: "other_object",
    5: "bike",
}

CLASS_NAME_TO_ID = {name: idx for idx, name in CLASS_ID_TO_NAME.items()}

NUM_CLASSES = len(CLASS_ID_TO_NAME)

