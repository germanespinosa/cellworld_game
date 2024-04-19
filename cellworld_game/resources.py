import os


class Resources(object):
    path = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def file(file_name: str) -> str:
        return os.path.join(Resources.path, "files", file_name)
