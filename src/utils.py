import numpy
class RLECoder():
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
        
        #print(len(starts))
        #print(len(ends))
        
        for index in range(len(starts)):
            result.append(starts[index])
            result.append(lens[index])
        #throw_error()
        
        '''
        for pixel in binary_image.flatten():

            if pixel > 0.5:
                if start is None:
                    start = index
                    length = 1

                else:
                    length += 1
            else:
                if start is not None:
                    result.append(start + 1)
                    result.append(length)
                    start = None
                    length = None
            index += 1
            
        if start is not None:
            result.append(start + 1)
            result.append(length)
        '''
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
