import tarfile

def extract_tar(file: str):
    """Extract tar file to folder"""
    try:
        tar = tarfile.open(file)
        tar.extractall()
        tar.close()
    except FileNotFoundError as err:
        print(err)
