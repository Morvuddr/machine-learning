class Entity:
    def __init__(self, target_value, attributes, normalized_attributes=None, class_number=-1, class_list=None):
        if class_list is None:
            class_list = []
        if normalized_attributes is None:
            normalized_attributes = []
        self.target_value = target_value
        self.attributes = attributes
        self.class_number = class_number
        self.normalized_attributes = normalized_attributes
        self.class_list = class_list
