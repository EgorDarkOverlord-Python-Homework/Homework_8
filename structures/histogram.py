import decoders.decoder
import encoders.encoder

"""
Стратегия (Strategy)
"""

class Hist():
    @classmethod
    def read(cls, file_path):
        ext = file_path.rsplit('.', 1)[-1]
        if ext == 'bin':
            decoder = decoders.decoder.BinHistDecoder
        elif ext == 'txt':
            decoder = decoders.decoder.TxtHistDecoder
        elif ext == 'json':
            decoder = decoders.decoder.JsonHistDecoder
        elif ext == 'csv':
            decoder = decoders.decoder.CsvHistDecoder
        elif ext in ('png', 'jpg', 'jpeg', 'bmp'):
            decoder = decoders.decoder.ImageHistDecoder
        else:
            raise RuntimeError('Невозможно получить данные %s' % file_path)
        data = decoder.decode(file_path)
        return cls(data)

    @classmethod
    def write(cls, filename):
        pass

    def __init__(self, data):
        self._data = data