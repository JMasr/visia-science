import mimetypes
import os
import pickle

import librosa
import numpy as np

from visia.responses.basic_responses import BasicResponse


def identify_file_type(file_path: str) -> str:
    """
    Identify the type of the file given its path. Possible types are: audio, video, text or unknown.

    :param file_path: Path to the file.
    :type file_path: str
    :return: Type of the file.
    :rtype: str

    """
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        major_type, _ = mime_type.split('/')
        return major_type.lower()
    else:
        return "unknown"


def get_file_format(file_path: str) -> str:
    """
    Extrae el formato (extensión) de un archivo.
    :param file_path: Ruta del archivo.
    :return: Extensión del archivo o una cadena vacía si no tiene extensión.
    """
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower()


def read_audio(audio_path: str) -> [np.ndarray, int]:
    """
    The code above implements SAD using the ´librosa.effects.split()´ function with a threshold of top_db=30, which
     separates audio regions where the amplitude is lower than the threshold.
     The pre-emphasis filter is applied using the ´librosa.effects.preemphasis()´ function with a coefficient of 0.97.
     This filter emphasizes the high-frequency components of the audio signal,
     which can improve the quality of the speech signal.
     Finally, the code normalizes the audio signal to have maximum amplitude of 1.

    :param audio_path: Path to the audio file.
    :type audio_path: str
    :return: Audio samples.
    :rtype: np.ndarray
    """
    # load the audio file
    s, sr = librosa.load(audio_path, sr=None, mono=True)
    # split the audio file into speech segments
    speech_indices = librosa.effects.split(s, top_db=30)
    s = np.concatenate([s[start:end] for start, end in speech_indices])
    # apply a pre-emphasis filter
    s = librosa.effects.preemphasis(s, coef=0.97)
    # normalize
    s /= np.max(np.abs(s))

    return s, sr


def save_file(file_path: str, data: bytes):
    """
    Save a file to disk as a pickle file.

    :param file_path: Path to the file.
    :type file_path: str
    :param data: File data.
    :type data: bytes
    """
    try:
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
        return BasicResponse(success=True, status_code=200, message="File saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}")
        return BasicResponse(success=False, status_code=500, message=f"Error saving file: {e}.")


def load_file(file_path: str) -> [bytes, BasicResponse]:
    """
    Load a file from disk.

    :param file_path: Path to the file.
    :type file_path: str
    :return: File data or a response with the error message.
    :rtype: bytes or BasicResponse
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return BasicResponse(success=False, status_code=500, message=f"Error loading file: {e}.")
