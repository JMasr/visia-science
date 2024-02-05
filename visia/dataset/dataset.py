from enum import Enum
from typing import Optional, Annotated, Any

import os
import cv2
import librosa
import numpy as np
import pandas as pd
import textgrids
from bson import ObjectId
from pydantic import BaseModel, FilePath, Field, ConfigDict

import visia.pydantic_numpy.dtype as pnd
from visia.files import identify_file_type, get_file_format, read_audio
from visia.pydantic_numpy.ndarray import NDArray


class DataTypes(Enum):
    """
    Enum class with the possible types of logs.
    """

    TEXT: int = 1
    AUDIO: int = 10
    VIDEO: int = 11
    OTHER: int = 99

    def __str__(self):
        """
        Return the name of the enum for a more human-readable log.

        :return: Name of the enum.
        :rtype: str
        """
        return self.name


class BaseStream(BaseModel):
    id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_]+$")] = None
    version: Annotated[str, Field(pattern=r"^[a-zA-Z0-9.]+$")] = "v.0.0"
    data: Any = None

    @classmethod
    def get_data_metadata(cls):
        class_attr = cls.__annotations__
        return {k: v for k, v in class_attr.items() if k != "data"}


class AudioStream(BaseStream):
    """
    Class to store the information of an audio stream.

    :param audio_format: Format of the audio.
    :type audio_format: str
    :param sample_rate: Sample rate of the audio.
    :type sample_rate: int
    :param duration: Duration of the audio.
    :type duration: float
    :param samples: Samples of the audio.
    :type samples: np.ndarray

    """
    audio_format: Annotated[str, Field(pattern=r"^[a-zA-Z0-9]+$")]
    sample_rate: Annotated[int, Field(strict=True, gt=0)]
    duration: Annotated[float, Field(strict=True, gt=0)]

    @classmethod
    def from_file_path(cls, file_path: str) -> "AudioStream":
        """
        Create a AudioStream object from the paths to the audio file.

        :param file_path: Path to the audio file.
        :type file_path: str
        """

        print(f"Reading audio file: {file_path}")
        s, sr = read_audio(file_path)

        id_media = os.path.basename(file_path).split(".")[0]
        audio_format = get_file_format(file_path)[1:]
        return cls(id=id_media,
                   audio_format=audio_format,
                   sample_rate=sr,
                   channels=1,
                   duration=librosa.get_duration(y=s, sr=sr),
                   data=s)


class VideoStream(BaseStream):
    """
    Class to store the information of a video stream.

    :param video_format: Format of the video.
    :type video_format: str
    :param duration: Duration of the video.
    :type duration: float
    :param frames: Frames of the video.
    :type frames: np.ndarray
    :param frame_rate: Frame rate of the video.
    :type frame_rate: int
    """
    video_format: Annotated[str, Field(pattern=r"^[a-zA-Z0-9]+$")]
    frames: NDArray[pnd.float32]
    duration: Annotated[float, Field(strict=True, gt=0)]
    frame_rate: Annotated[int, Field(strict=True, gt=0)]

    @classmethod
    def from_file_path(cls, file_path: str) -> "VideoStream":
        """
        Create a VideoStream object from the paths to the video file.

        :param file_path: Path to the video file.
        :type file_path: str
        """
        video = cv2.VideoCapture(file_path)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                frames.append(frame)
            else:
                break
        video.release()
        return cls(id=os.path.basename(file_path).split(".")[0],
                   video_format=get_file_format(file_path),
                   codec=video.getBackendName(),
                   duration=video.get(cv2.CAP_PROP_POS_MSEC),
                   frame_rate=video.get(cv2.CAP_PROP_FPS),
                   data=np.array(frames),)


class TextStream(BaseStream):
    """
    Class to store the information of a text event (text file + text stream).

    :param id: ID of the text file.
    :type id: str
    :param text_stream: Text stream.
    :type text_stream: pd.DataFrame

    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text_stream: Optional[pd.DataFrame] = None

    @classmethod
    def from_file_path(cls, file_path: str) -> "TextStream":

        # Get the file name and extension
        file_name = os.path.basename(file_path).split(".")[0]
        file_ext = os.path.basename(file_path).split(".")[1]

        # Read the file
        if file_ext == "csv":
            df_txt = pd.read_csv(file_path)
        elif file_ext == "json":
            df_txt = pd.read_json(file_path)
        elif file_ext == "TextGrid":
            grid_txt = textgrids.TextGrid(file_path)
            df_txt = pd.DataFrame(grid_txt["words"])

        else:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            df_txt = pd.DataFrame(lines)

        return cls(id=file_name,
                   text_stream=df_txt)


class AudioEvent(BaseModel):
    """
    Class to store the information of an audio event (audio file + audio stream).

    :param audio_file: Path to the audio file.
    :type audio_file: str
    :param audio_stream: Audio stream.
    :type audio_stream: AudioStream

    """
    audio_file: FilePath
    audio_stream: Optional[AudioStream] = None


class VideoEvent(BaseModel):
    """
    Class to store the information of a video event (video file + video stream).

    :param video_file: Path to the video file.
    :type video_file: str
    :param video_stream: Video stream.
    :type video_stream: VideoStream

    """
    video_file: FilePath
    video_stream: Optional[VideoStream] = None


class MediaFile(BaseModel):
    """
    Class to store the information of a media file.

    :param id: ID of the media file.
    :type id: str
    :param audio_event: Audio event (audio file + audio stream).
    :type audio_event: AudioEvent
    :param video_event: Video event (video file + video stream).
    :type video_event: VideoEvent
    :param metadata_events: Metadata events (text file + text stream).
    :type metadata_events: dict

    """
    id: str = Annotated[str, Field(pattern=r"^[a-zA-Z0-9]+$")]

    audio_event: AudioEvent = None
    video_event: Optional[VideoEvent] = None
    metadata_events: Optional[dict] = None

    @classmethod
    def from_file_paths(cls, file_path: str, id_media: str = None) -> "MediaFile":
        """
        Create a media file object from the paths to the files.

        :param file_path: Path to the file.
        :type file_path: str
        :param id_media: ID of the media file.
        :type id_media: str
        """

        if id_media is None:
            file_name = os.path.basename(file_path).split(".")[0]
            id_media = file_name

        file_type = identify_file_type(file_path)
        if file_type == "unknown":
            raise ValueError(f"Error file type *{file_type}* not recognized")
        else:
            audio_stream = AudioStream.from_file_path(file_path)

            if file_type == "audio":
                return cls(id=id_media,
                           audio_event=AudioEvent(audio_file=file_path,
                                                  audio_stream=audio_stream))
            elif file_type == "video":
                video_stream = VideoStream.from_file_path(file_path)

                return cls(id=id_media,
                           audio_event=AudioEvent(audio_file=file_path, audio_stream=audio_stream),
                           video_event=VideoEvent(video_file=file_path, video_stream=video_stream))

    def add_metadata(self, event_type: str, metadata_name: str, metadata_content: TextStream) -> None:
        """
        Add new metadata to the object. The metadata will be added to the specified event type.

        :param event_type: Type of event to add the metadata to.
        :type event_type: str
        :param metadata_name: Name of the metadata to add.
        :type metadata_name: str
        :param metadata_content: Content of the metadata.
        :type metadata_content: str
        :return: None

        """
        if event_type == "metadata":
            if self.metadata_events is None:
                self.metadata_events = {}

            self.metadata_events[metadata_name] = metadata_content
        else:
            raise ValueError(f"Error event type *{event_type}* not recognized")
