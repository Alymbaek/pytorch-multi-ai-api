from .db import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Text




class Gtzan(Base):
    __tablename__ = 'audio_gtzan'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, unique=True)
    label: Mapped[str] = mapped_column(String)
    predict: Mapped[str] = mapped_column(String)

class SpeechCommands(Base):
    __tablename__ = 'audio_speech_commands'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, unique=True)
    label: Mapped[str] = mapped_column(String)
    predict: Mapped[str] = mapped_column(String)

class Urban(Base):
    __tablename__ = 'audio_urban'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, unique=True)
    label: Mapped[str] = mapped_column(String)
    predict: Mapped[str] = mapped_column(String)

#image class

class Cifar(Base):
    __tablename__ = 'image_cifar'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, unique=True)
    label: Mapped[str] = mapped_column(String)
    predict: Mapped[str] = mapped_column(String)

class FashionMnist(Base):
    __tablename__ = 'image_fashion_mnist'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, unique=True)
    label: Mapped[str] = mapped_column(String)
    predict: Mapped[str] = mapped_column(String)

class Mnist(Base):
    __tablename__ = 'image_mnist'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, unique=True)
    label: Mapped[str] = mapped_column(String)
    predict: Mapped[str] = mapped_column(String)

# Text class

class Text_ag_news(Base):
    __tablename__ = 'text_ag_news'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, unique=True)
    label: Mapped[str] = mapped_column(String)
    predict: Mapped[str] = mapped_column(String)