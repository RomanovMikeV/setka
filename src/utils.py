import numpy

def torch_image_to_numpy(image):
    res = image.clone()
    res = res.swapaxes(0, 1).swapaxes(1, 2)
    return res


def numpy_image_to_torch(image):
    res = image.clone()
    res = res.swapaxes(2, 1).swapaxes(1, 0)
    return res


class RLECoder():
    '''
    Class implements Run-Length encoder/decoder.
    Note that the indexing starts from 1.
    '''
    def __init__(self):
        pass
    
    def encode(self, binary_image):
        start = None
        length = None
        
        result = [*binary_image.shape]

        index = 0
        
#             line = line[2:]
        b_image = binary_image.flatten()
        b_image = numpy.pad(b_image, [[1, 1]], mode='constant')
        
        shifts = b_image[:-1] - b_image[1:]
        
        starts = numpy.where(shifts < -0.5)[0]
        ends = numpy.where(shifts > 0.5)[0]
        
        
        lens = ends - starts
        starts += 1
        
        for index in range(len(starts)):
            result.append(starts[index])
            result.append(lens[index])
            
        return result

    def decode(self, rle):
        shape = rle[:2]
        code = rle[2:]
        image = numpy.zeros(shape)
        image = image.flatten()

        for index in range(0, len(code), 2):
            start = code[index] - 1
            length = code[index + 1]
            image[start:start+length] = 1

        image = image.reshape(shape)
        return image


def correct_coordinate(x, left_pad, right_pad):
    '''
    Function corrects the relative coordinate when the
    sequence was padded from the left and from the right.
    '''
    return (x - left_pad) / (1.0 - left_pad - right_pad)


def pad_image(img, crops):
    '''
    Function recomputes the image and the objects masks
    when the image is padded.
    '''
    
    left_crop   = crops[0]
    top_crop    = crops[1]
    right_crop  = crops[2]
    bottom_crop = crops[3]
    
    
    left_crop = left_crop * (img.shape[0] - 1)
    right_crop = (1.0 - right_crop) * (img.shape[0] - 1)
    top_crop = top_crop * (img.shape[1] - 1)
    bottom_crop = (1.0 - bottom_crop) * (img.shape[1] - 1)
        
    img = img[round(left_crop):round(right_crop) + 1, 
              round(top_crop):round(bottom_crop) + 1, :]
    
    return img


def pad_mask(masks, crops):
    '''
    Function recomputes mask borders whent the image
    is cropped
    '''
    
    left_crop   = crops[0]
    top_crop    = crops[1]
    right_crop  = crops[2]
    bottom_crop = crops[3]
    
    actual_masks = []
    for mask in masks:
        mask[0][0] = correct_coordinate(
            mask[0][0], left_crop, right_crop)
        mask[0][1] = correct_coordinate(
            mask[0][1], top_crop, bottom_crop)
        mask[0][2] = correct_coordinate(
            mask[0][2], left_crop, right_crop)
        mask[0][3] = correct_coordinate(
            mask[0][3], top_crop, bottom_crop)
            
        if not (mask[0][0] >= 1.0 or mask[0][1] >= 1.0 or
                mask[0][2] <= 0.0 or mask[0][3] <= 0.0):
            actual_masks.append(mask)
    
    return actual_masks


def correct_masks(masks):
    '''
    Function corrects masks in case the mask is
    getting outside of the image boundaries.
    '''
    
    new_masks = []
    for index in range(len(masks)):
        mask = masks[index]
        if not (mask[0][0] >= 0.0 and
                mask[0][1] >= 0.0 and
                mask[0][2] <= 1.0 and
                mask[0][3] <= 1.0):
            
            new_mask = [deepcopy(mask[0]), deepcopy(mask[1])]
        
            if mask[0][0] < 0.0:
                new_mask[0][0] = 0.0
        
            if mask[0][1] < 0.0:
                new_mask[0][1] = 0.0
        
            if mask[0][2] > 1.0:
                new_mask[0][2] = 1.0
        
            if mask[0][3] > 1.0:
                new_mask[0][3] = 1.0
                
            old_mask_width = mask[0][2] - mask[0][0]
            old_mask_height = mask[0][3] - mask[0][1]
#             new_mask[1]
            
            new_x_start = round(
                (new_mask[0][0] - mask[0][0]) / 
                old_mask_width * (mask[1].shape[0] - 1))
            new_y_start = round(
                (new_mask[0][1] - mask[0][1]) / 
                old_mask_height * (mask[1].shape[1]- 1))
            new_x_end = round(
                (new_mask[0][2] - mask[0][0]) / 
                old_mask_width * (mask[1].shape[0] - 1))
            new_y_end = round(
                (new_mask[0][3] - mask[0][1]) / 
                old_mask_height * (mask[1].shape[1] - 1))
            
            new_mask[1] = mask[1][
                new_x_start:new_x_end + 1, 
                new_y_start:new_y_end + 1]
            
            if new_mask[1].shape[0] > 1 and new_mask[1].shape[1] > 1:
                new_masks.append(new_mask)
        else:
            new_masks.append(mask)
        
    return new_masks


