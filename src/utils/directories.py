import os


def get_parent_directory():
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    return parent_dir
