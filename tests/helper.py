from os import PathLike


def what(file):
    tests = []

    def test_jpeg(h):
        """JPEG data with JFIF or Exif markers; and raw JPEG"""
        if h[6:10] in (b'JFIF', b'Exif'):
            return 'jpeg'
        elif h[:4] == b'\xff\xd8\xff\xdb':
            return 'jpeg'

    tests.append(test_jpeg)

    def test_png(h):
        if h.startswith(b'\211PNG\r\n\032\n'):
            return 'png'

    tests.append(test_png)

    f = None
    try:
        if isinstance(file, (str, PathLike)):
            f = open(file, 'rb')
            h = f.read(32)
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)
        for tf in tests:
            res = tf(h)
            if res:
                return res
    finally:
        if f:
            f.close()
    return None
