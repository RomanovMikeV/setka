import os
import skimage.io

class OpenImages():
    def __init__(self, oi_path):
        self.path = oi_path
        
        self.class_id_to_output = {}
        self.output_to_class_id = []
        self.class_id_to_name = {}
        
        output_index = 0
        with open(os.path.join(self.path, 'index/classes-trainable.txt')) as fin:
            for line in fin:
                class_id = line.strip()
                self.class_id_to_output[class_id] = output_index
                self.output_to_class_id.append(class_id)
                output_index += 1
        
        with open(os.path.join(self.path, 'index/class-descriptions.csv')) as fin:
            for line in fin:
                line = line.strip().split(',')
                class_id = line[0]
                class_name = ','.join(line[1:])
                self.class_id_to_name[class_id] = class_name
#                 throw_error()
        
        train_human_labels = os.path.join(
            self.path, 'index/train/annotations-human.csv')
        test_human_labels = os.path.join(
            self.path, 'index/test/annotations-human.csv')
        valid_human_labels = os.path.join(
            self.path, 'index/validation/annotations-human.csv')
        
        labels_files = [train_human_labels, 
                        test_human_labels,
                        valid_human_labels]
        
        self.image_labels = {}
            
        indice_sets = []
        
        for labels_file in labels_files:
            with open(labels_file) as fin:
                indice_set = {}
                next(fin)
                for line in fin:
                    line = line.strip().split(',')
                    
                    img_id = line[0]
                    class_id = line[2]
                    score = float(line[3])
                    
                    if score > 0.5:
                        if class_id in self.class_id_to_output:
                                
                            indice_set[img_id] = True
                            
                            if img_id not in self.image_labels:
                                self.image_labels[img_id] = []
                            
                            self.image_labels[img_id].append(class_id)
                                
                indice_sets.append(list(indice_set.keys()))
        
        self.train_images = indice_sets[0]
        self.test_images = indice_sets[1]
        self.valid_images = indice_sets[2]
        
    def load_image_test(self, test_index):
        image_id = self.test_images[test_index]
        img = skimage.io.imread(
                os.path.join(self.path, 'test', image_id + '.jpg'))
        target = self.image_labels[image_id]
        return img, target
    
    def load_image_train(self, train_index):
        image_id = self.train_images[test_index]
        img = skimage.io.imread(
                os.path.join(self.path, 'train', image_id + '.jpg'))
        target = self.image_labels[image_id]
        return img, target
    
    def load_image_valid(self, valid_index):
        image_id = self.valid_images[test_index]
        img = skimage.io.imread(
                os.path.join(self.path, 'validation', image_id + '.jpg'))
        target = self.image_labels[image_id]
        return img, target