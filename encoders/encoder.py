import struct
import csv
import json
import cv2

class HistEncoder:
    @staticmethod
    def encode(file_path, data):
        raise NotImplementedError()
    
class BinHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        with open(file_path, "wb") as file:
            binary_data = b''.join(struct.pack('f', f) for f in list(data.values()))
            file.write(binary_data)

class CsvHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data.items():
                writer.writerow([key, value])

class TxtHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        with open(file_path, "w") as file:
            for key, value in data.items():
                file.write('{} {} \n'.format(key,value))

class JsonHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        reform_data = {'keys':list(data.keys()),
                   'values':list(data.values())}

        with open(file_path, 'w') as file:
            json.dump(reform_data, file)

class ImageHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        cv2.imwrite(file_path, data)

